import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "ERROR: PyTorch (torch) is not installed.\n"
        "This project requires PyTorch even on CPU.\n\n"
        "Install (recommended, conda):\n"
        "  conda install -c pytorch pytorch cpuonly -y\n\n"
        "Or install with pip (CPU wheels):\n"
        "  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
    ) from e


from common import compute_metrics, load_prepared_split, seed_everything, write_json
from models import encode_seq, reverse_complement


class PairedDataset(Dataset):
    """Dataset for WT/MT/Δ targets with optional reverse-complement augmentation."""

    def __init__(self, df: pd.DataFrame, rc_aug: bool = False, rc_prob: float = 0.5):
        self.seqs = df["sequence"].astype(str).tolist()
        self.y_wt = pd.to_numeric(df.get("log2_WT"), errors="coerce").values.astype(np.float32)
        self.y_mt = pd.to_numeric(df.get("log2_MT"), errors="coerce").values.astype(np.float32)
        self.y_delta = pd.to_numeric(df.get("delta"), errors="coerce").values.astype(np.float32)
        self.y_mean = pd.to_numeric(df.get("mean"), errors="coerce").values.astype(np.float32)

        def _col(name: str, default: float = 1.0):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float32)
            return np.full(len(df), default, dtype=np.float32)

        self.w_wt = _col("w_int")
        self.w_mt = _col("w_epi")
        self.w_delta = _col("w_delta")
        self.w_mean = _col("w_mean")

        if "sample_weight" in df.columns and ("w_delta" not in df.columns):
            sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(np.float32)
            self.w_wt = sw
            self.w_mt = sw
            self.w_delta = sw
            self.w_mean = sw

        self.rc_aug = bool(rc_aug)
        self.rc_prob = float(rc_prob)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        if self.rc_aug and (np.random.rand() < self.rc_prob):
            s = reverse_complement(s)
        x = encode_seq(s)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_wt[idx]),
            torch.tensor(self.y_mt[idx]),
            torch.tensor(self.y_delta[idx]),
            torch.tensor(self.y_mean[idx]),
            torch.tensor(self.w_wt[idx]),
            torch.tensor(self.w_mt[idx]),
            torch.tensor(self.w_delta[idx]),
            torch.tensor(self.w_mean[idx]),
        )


class SmallCNN3Head(nn.Module):
    """CPU-friendly CNN trunk with three regression heads: WT, MT, and Δ."""

    def __init__(self, vocab_size: int = 5, d: int = 32, dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_wt = nn.Linear(128, 1)
        self.head_mt = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor):
        h = self.emb(x_tok).transpose(1, 2)  # (B, d, L)
        h1 = self.act(self.conv1(h))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h1))
        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h2, dim=-1)
        p3 = torch.amax(h3, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        feat = self.trunk(feat)
        p_wt = self.head_wt(feat).squeeze(-1)
        p_mt = self.head_mt(feat).squeeze(-1)
        p_delta = self.head_delta(feat).squeeze(-1)
        return p_wt, p_mt, p_delta


class EMA:
    """Simple exponential moving average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self._backup = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            self._backup[k] = sd[k].detach().clone()
            sd[k].copy_(v)

    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    y_wt, y_mt, y_delta, y_mean = [], [], [], []
    p_wt, p_mt, p_delta = [], [], []
    for (x, wt, mt, delt, mean, *_w) in loader:
        x = x.to(device)
        pw, pm, pd = model(x)
        y_wt.append(wt.numpy()); y_mt.append(mt.numpy()); y_delta.append(delt.numpy()); y_mean.append(mean.numpy())
        p_wt.append(pw.cpu().numpy()); p_mt.append(pm.cpu().numpy()); p_delta.append(pd.cpu().numpy())
    y_wt = np.concatenate(y_wt)
    y_mt = np.concatenate(y_mt)
    y_delta = np.concatenate(y_delta)
    y_mean = np.concatenate(y_mean)
    p_wt = np.concatenate(p_wt)
    p_mt = np.concatenate(p_mt)
    p_delta = np.concatenate(p_delta)
    p_delta_derived = p_wt - p_mt
    p_mean_derived = 0.5 * (p_wt + p_mt)
    return {
        "y_wt": y_wt,
        "y_mt": y_mt,
        "y_delta": y_delta,
        "y_mean": y_mean,
        "p_wt": p_wt,
        "p_mt": p_mt,
        "p_delta": p_delta,
        "p_delta_derived": p_delta_derived,
        "p_mean_derived": p_mean_derived,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--patience", type=int, default=22)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--rc_aug", action="store_true")
    ap.add_argument("--rc_prob", type=float, default=0.5)

    ap.add_argument("--loss", default="huber", choices=["huber", "mse"])
    ap.add_argument("--huber_beta", type=float, default=0.5)
    ap.add_argument("--w_wt", type=float, default=1.0)
    ap.add_argument("--w_mt", type=float, default=1.0)
    ap.add_argument("--w_delta", type=float, default=1.0)
    ap.add_argument("--w_mean", type=float, default=0.0)
    ap.add_argument("--delta_consistency_lambda", type=float, default=0.2,
                    help="Penalty for (Δ_pred - (WT_pred - MT_pred))^2. 0 disables.")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--stop_on", default="delta", choices=["delta", "wt", "mt", "avg_wt_mt", "delta_derived"])
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_prepared_split(str(Path(args.prepared_dir) / "train.tsv"))
    val_df = load_prepared_split(str(Path(args.prepared_dir) / "val.tsv"))
    test_df = load_prepared_split(str(Path(args.prepared_dir) / "test.tsv"))

    for col in ["log2_WT", "log2_MT", "delta", "mean"]:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column '{col}' in prepared TSV.")

    train_ds = PairedDataset(train_df, rc_aug=args.rc_aug, rc_prob=args.rc_prob)
    val_ds = PairedDataset(val_df, rc_aug=False)
    test_ds = PairedDataset(test_df, rc_aug=False)

    device = torch.device("cpu")
    model = SmallCNN3Head(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    if args.loss == "huber":
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=float(args.huber_beta))
    else:
        loss_fn = nn.MSELoss(reduction="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def val_score(pack: Dict[str, np.ndarray]) -> Dict[str, float]:
        m_wt = compute_metrics(pack["y_wt"], pack["p_wt"])
        m_mt = compute_metrics(pack["y_mt"], pack["p_mt"])
        m_delta = compute_metrics(pack["y_delta"], pack["p_delta"])
        m_dd = compute_metrics(pack["y_delta"], pack["p_delta_derived"])
        return {
            "val_wt_pearson": float(m_wt["pearson"]),
            "val_mt_pearson": float(m_mt["pearson"]),
            "val_delta_pearson": float(m_delta["pearson"]),
            "val_delta_derived_pearson": float(m_dd["pearson"]),
        }

    def score(vm: Dict[str, float]) -> float:
        if args.stop_on == "wt":
            return vm["val_wt_pearson"]
        if args.stop_on == "mt":
            return vm["val_mt_pearson"]
        if args.stop_on == "avg_wt_mt":
            return 0.5 * (vm["val_wt_pearson"] + vm["val_mt_pearson"])
        if args.stop_on == "delta_derived":
            return vm["val_delta_derived_pearson"]
        return vm["val_delta_pearson"]

    best = -1e9
    bad = 0
    best_path = out_dir / "cnn_wt_mt_delta3head_best.pt"
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for (x, y_wt, y_mt, y_delta, y_mean, w_wt, w_mt, w_delta, w_mean) in train_loader:
            x = x.to(device)
            y_wt = y_wt.to(device); y_mt = y_mt.to(device)
            y_delta = y_delta.to(device); y_mean = y_mean.to(device)
            w_wt = w_wt.to(device); w_mt = w_mt.to(device)
            w_delta = w_delta.to(device); w_mean = w_mean.to(device)

            opt.zero_grad(set_to_none=True)
            p_wt, p_mt, p_delta = model(x)

            l_wt = loss_fn(p_wt, y_wt) * w_wt
            l_mt = loss_fn(p_mt, y_mt) * w_mt
            l_delta = loss_fn(p_delta, y_delta) * w_delta
            loss = args.w_wt * l_wt + args.w_mt * l_mt + args.w_delta * l_delta

            if args.w_mean and args.w_mean > 0:
                p_mean = 0.5 * (p_wt + p_mt)
                l_mean = loss_fn(p_mean, y_mean) * w_mean
                loss = loss + args.w_mean * l_mean

            if args.delta_consistency_lambda and args.delta_consistency_lambda > 0:
                cons = (p_delta - (p_wt - p_mt)) ** 2
                loss = loss + float(args.delta_consistency_lambda) * cons

            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)

            total += float(loss.item()) * len(x)
            n += len(x)

        # validation
        if ema is not None:
            ema.apply_to(model)
        vpack = predict_all(model, val_loader, device)
        if ema is not None:
            ema.restore(model)

        vm = val_score(vpack)
        sc = score(vm)
        hist.append({"epoch": ep, "train_loss": total / max(1, n), **vm})

        if sc > best + 1e-6:
            best = sc
            bad = 0
            # save raw params (not EMA) + metadata so downstream scripts can load as state_dict
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                break

    # evaluate best
    model.load_state_dict(torch.load(best_path, map_location=device))

    vpack = predict_all(model, val_loader, device)
    tpack = predict_all(model, test_loader, device)

    m_wt = compute_metrics(tpack["y_wt"], tpack["p_wt"])
    m_mt = compute_metrics(tpack["y_mt"], tpack["p_mt"])
    m_delta = compute_metrics(tpack["y_delta"], tpack["p_delta"])
    m_dd = compute_metrics(tpack["y_delta"], tpack["p_delta_derived"])

    # write val predictions (for fusion)
    val_out = val_df[["element_id", "sequence", "log2_WT", "log2_MT", "delta", "mean"]].copy()
    val_out = val_out.rename(columns={"log2_WT": "y_int", "log2_MT": "y_epi", "delta": "y_delta", "mean": "y_mean"})
    val_out["pred_int"] = vpack["p_wt"]
    val_out["pred_epi"] = vpack["p_mt"]
    val_out["pred_delta"] = vpack["p_delta"]
    val_out["pred_delta_derived"] = vpack["p_delta_derived"]
    val_out["pred_mean_derived"] = vpack["p_mean_derived"]
    val_out.to_csv(out_dir / "cnn_wt_mt_delta3head.val_predictions.tsv", sep="\t", index=False)

    test_out = test_df[["element_id", "sequence", "log2_WT", "log2_MT", "delta", "mean"]].copy()
    test_out = test_out.rename(columns={"log2_WT": "y_int", "log2_MT": "y_epi", "delta": "y_delta", "mean": "y_mean"})
    test_out["pred_int"] = tpack["p_wt"]
    test_out["pred_epi"] = tpack["p_mt"]
    test_out["pred_delta"] = tpack["p_delta"]
    test_out["pred_delta_derived"] = tpack["p_delta_derived"]
    test_out["pred_mean_derived"] = tpack["p_mean_derived"]
    test_out.to_csv(out_dir / "cnn_wt_mt_delta3head.test_predictions.tsv", sep="\t", index=False)

    metrics = {
        "model": "cnn_wt_mt_delta3head",
        "seed": int(args.seed),
        "best_val_score": float(best),
        "test_wt_pearson": float(m_wt["pearson"]),
        "test_mt_pearson": float(m_mt["pearson"]),
        "test_delta_pearson": float(m_delta["pearson"]),
        "test_delta_derived_pearson": float(m_dd["pearson"]),
        "test_wt_spearman": float(m_wt["spearman"]),
        "test_mt_spearman": float(m_mt["spearman"]),
        "test_delta_spearman": float(m_delta["spearman"]),
        "test_delta_derived_spearman": float(m_dd["spearman"]),
        "n_test": int(len(test_out)),
        "config": vars(args),
    }
    write_json(metrics, str(out_dir / "cnn_wt_mt_delta3head.metrics.json"))
    write_json(hist, str(out_dir / "cnn_wt_mt_delta3head.history.json"))

    print("Wrote:", out_dir)
    print(metrics)


if __name__ == "__main__":
    main()
