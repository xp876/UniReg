
import argparse
from pathlib import Path
from typing import Dict, Tuple

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
from models import SmallCNNMeanDelta, encode_seq, reverse_complement, derive_int_epi


class PairedDatasetWTMT(Dataset):
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

        # noise-aware weights
        self.w_wt = _col("w_int")
        self.w_mt = _col("w_epi")
        self.w_delta = _col("w_delta")
        self.w_mean = _col("w_mean")

        # fallback
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


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    y_wt, y_mt, y_delta, y_mean = [], [], [], []
    p_wt, p_mt, p_delta, p_mean = [], [], [], []
    with torch.no_grad():
        for (x, wt, mt, delt, mean, *_w) in loader:
            x = x.to(device)
            pm, pd = model(x)
            # derive WT/MT in log2 space
            pi, pe = derive_int_epi(pm.cpu().numpy(), pd.cpu().numpy())
            y_wt.append(wt.numpy()); y_mt.append(mt.numpy())
            y_delta.append(delt.numpy()); y_mean.append(mean.numpy())
            p_wt.append(pi); p_mt.append(pe)
            p_delta.append(pd.cpu().numpy()); p_mean.append(pm.cpu().numpy())
    return {
        "y_wt": np.concatenate(y_wt),
        "y_mt": np.concatenate(y_mt),
        "y_delta": np.concatenate(y_delta),
        "y_mean": np.concatenate(y_mean),
        "p_wt": np.concatenate(p_wt),
        "p_mt": np.concatenate(p_mt),
        "p_delta": np.concatenate(p_delta),
        "p_mean": np.concatenate(p_mean),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--patience", type=int, default=18)
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
    ap.add_argument("--w_mean", type=float, default=0.0, help="Optional mean supervision weight")
    ap.add_argument("--delta_clip_q", type=float, default=0.0, help="Winsorize delta by quantiles (e.g., 0.01). 0 disables.")
    ap.add_argument("--stop_on", default="delta", choices=["delta", "wt", "mt", "avg_wt_mt"])

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

    # optional winsorize delta targets (train only)
    if args.delta_clip_q and args.delta_clip_q > 0:
        q = float(args.delta_clip_q)
        lo = float(np.quantile(train_df["delta"].values.astype(float), q))
        hi = float(np.quantile(train_df["delta"].values.astype(float), 1 - q))
        train_df = train_df.copy()
        train_df["delta"] = np.clip(train_df["delta"].values.astype(float), lo, hi)
        # keep consistency for mean/wt/mt by leaving them unchanged; we are intentionally robustifying delta loss only.

    train_ds = PairedDatasetWTMT(train_df, rc_aug=args.rc_aug, rc_prob=args.rc_prob)
    val_ds = PairedDatasetWTMT(val_df, rc_aug=False)
    test_ds = PairedDatasetWTMT(test_df, rc_aug=False)

    device = torch.device("cpu")
    model = SmallCNNMeanDelta(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss == "huber":
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=float(args.huber_beta))
    else:
        loss_fn = nn.MSELoss(reduction="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def val_metrics(pack: Dict[str, np.ndarray]) -> Dict[str, float]:
        m_wt = compute_metrics(pack["y_wt"], pack["p_wt"])
        m_mt = compute_metrics(pack["y_mt"], pack["p_mt"])
        m_delta = compute_metrics(pack["y_delta"], pack["p_delta"])
        m_mean = compute_metrics(pack["y_mean"], pack["p_mean"])
        return {
            "val_wt_pearson": float(m_wt["pearson"]),
            "val_mt_pearson": float(m_mt["pearson"]),
            "val_delta_pearson": float(m_delta["pearson"]),
            "val_mean_pearson": float(m_mean["pearson"]),
        }

    def score(vm: Dict[str, float]) -> float:
        if args.stop_on == "wt":
            return vm["val_wt_pearson"]
        if args.stop_on == "mt":
            return vm["val_mt_pearson"]
        if args.stop_on == "avg_wt_mt":
            return 0.5 * (vm["val_wt_pearson"] + vm["val_mt_pearson"])
        return vm["val_delta_pearson"]

    best = -1e9
    best_path = out_dir / "cnn_wt_mt_delta_best.pt"
    bad = 0
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
            p_mean, p_delta = model(x)
            # derive predictions
            p_wt = p_mean + 0.5 * p_delta
            p_mt = p_mean - 0.5 * p_delta

            l_wt = loss_fn(p_wt, y_wt) * w_wt
            l_mt = loss_fn(p_mt, y_mt) * w_mt
            l_delta = loss_fn(p_delta, y_delta) * w_delta
            loss = args.w_wt * l_wt + args.w_mt * l_mt + args.w_delta * l_delta

            if args.w_mean and args.w_mean > 0:
                l_mean = loss_fn(p_mean, y_mean) * w_mean
                loss = loss + args.w_mean * l_mean

            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(x)
            n += len(x)

        # val
        pack = predict_all(model, val_loader, device)
        vm = val_metrics(pack)
        sc = score(vm)
        hist.append({"epoch": ep, "train_loss": total / max(1, n), **vm})

        if sc > best + 1e-6:
            best = sc
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                break

    # load best + evaluate
    model.load_state_dict(torch.load(best_path, map_location=device))

    val_pack = predict_all(model, val_loader, device)
    pack = predict_all(model, test_loader, device)
    m_wt = compute_metrics(pack["y_wt"], pack["p_wt"])
    m_mt = compute_metrics(pack["y_mt"], pack["p_mt"])
    m_delta = compute_metrics(pack["y_delta"], pack["p_delta"])
    m_mean = compute_metrics(pack["y_mean"], pack["p_mean"])

    # write validation predictions (for stacking / fusion)
    val_out = val_df[["element_id", "sequence", "log2_WT", "log2_MT", "delta", "mean"]].copy()
    val_out = val_out.rename(columns={"log2_WT":"y_int", "log2_MT":"y_epi", "delta":"y_delta", "mean":"y_mean"})
    val_out["pred_int"] = val_pack["p_wt"]
    val_out["pred_epi"] = val_pack["p_mt"]
    val_out["pred_delta"] = val_pack["p_delta"]
    val_out["pred_mean"] = val_pack["p_mean"]
    val_out.to_csv(out_dir / "cnn_wt_mt_delta.val_predictions.tsv", sep="\t", index=False)


    # write predictions
    test_out = test_df[["element_id", "sequence", "log2_WT", "log2_MT", "delta", "mean"]].copy()
    test_out = test_out.rename(columns={"log2_WT":"y_int", "log2_MT":"y_epi", "delta":"y_delta", "mean":"y_mean"})
    test_out["pred_int"] = pack["p_wt"]
    test_out["pred_epi"] = pack["p_mt"]
    test_out["pred_delta"] = pack["p_delta"]
    test_out["pred_mean"] = pack["p_mean"]
    test_out.to_csv(out_dir / "cnn_wt_mt_delta.test_predictions.tsv", sep="\t", index=False)

    metrics = {
        "model": "cnn_wt_mt_delta",
        "seed": int(args.seed),
        "best_val_score": float(best),
        "test_wt_pearson": float(m_wt["pearson"]),
        "test_mt_pearson": float(m_mt["pearson"]),
        "test_delta_pearson": float(m_delta["pearson"]),
        "test_mean_pearson": float(m_mean["pearson"]),
        "test_wt_spearman": float(m_wt["spearman"]),
        "test_mt_spearman": float(m_mt["spearman"]),
        "test_delta_spearman": float(m_delta["spearman"]),
        "test_mean_spearman": float(m_mean["spearman"]),
        "n_test": int(len(test_out)),
    }
    write_json(metrics, str(out_dir / "cnn_wt_mt_delta.metrics.json"))
    write_json(hist, str(out_dir / "cnn_wt_mt_delta.history.json"))

    print("Wrote:", out_dir)
    print(metrics)

if __name__ == "__main__":
    main()
