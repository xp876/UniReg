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
from models import SmallCNNMeanDelta, encode_seq, reverse_complement, derive_int_epi


class PairedMeanDeltaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, rc_aug: bool = False, rc_prob: float = 0.5):
        self.seqs = df["sequence"].astype(str).tolist()

        # targets
        self.y_mean = pd.to_numeric(df.get("mean"), errors="coerce").values.astype(np.float32)
        self.y_delta = pd.to_numeric(df.get("delta"), errors="coerce").values.astype(np.float32)
        # optional absolute targets (for consistency loss)
        self.y_int = pd.to_numeric(df.get("log2_WT"), errors="coerce").values.astype(np.float32)
        self.y_epi = pd.to_numeric(df.get("log2_MT"), errors="coerce").values.astype(np.float32)

        # noise-aware weights (optional)
        def _col(name: str, default: float = 1.0):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float32)
            return np.full(len(df), default, dtype=np.float32)

        self.w_mean = _col("w_mean")
        self.w_delta = _col("w_delta")
        self.w_int = _col("w_int")
        self.w_epi = _col("w_epi")

        # fallback: if only sample_weight exists, use it for all
        if "sample_weight" in df.columns and ("w_mean" not in df.columns):
            sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(np.float32)
            self.w_mean = sw
            self.w_delta = sw
            self.w_int = sw
            self.w_epi = sw

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
            torch.tensor(self.y_mean[idx]),
            torch.tensor(self.y_delta[idx]),
            torch.tensor(self.y_int[idx]),
            torch.tensor(self.y_epi[idx]),
            torch.tensor(self.w_mean[idx]),
            torch.tensor(self.w_delta[idx]),
            torch.tensor(self.w_int[idx]),
            torch.tensor(self.w_epi[idx]),
        )


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    y_mean, y_delta, y_int, y_epi = [], [], [], []
    p_mean, p_delta, p_int, p_epi = [], [], [], []
    with torch.no_grad():
        for (x, ym, yd, yi, ye, *_w) in loader:
            x = x.to(device)
            pm, pd = model(x)
            pm = pm.cpu().numpy()
            pd = pd.cpu().numpy()
            pi, pe = derive_int_epi(pm, pd)

            y_mean.append(ym.numpy())
            y_delta.append(yd.numpy())
            y_int.append(yi.numpy())
            y_epi.append(ye.numpy())

            p_mean.append(pm)
            p_delta.append(pd)
            p_int.append(pi)
            p_epi.append(pe)

    out = {
        "y_mean": np.concatenate(y_mean),
        "y_delta": np.concatenate(y_delta),
        "y_int": np.concatenate(y_int),
        "y_epi": np.concatenate(y_epi),
        "p_mean": np.concatenate(p_mean),
        "p_delta": np.concatenate(p_delta),
        "p_int": np.concatenate(p_int),
        "p_epi": np.concatenate(p_epi),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--rc_aug", action="store_true")
    ap.add_argument("--rc_prob", type=float, default=0.5)

    ap.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse"])
    ap.add_argument("--w_mean", type=float, default=1.0, help="Loss weight for mean")
    ap.add_argument("--w_delta", type=float, default=1.0, help="Loss weight for delta")
    ap.add_argument("--consistency", type=float, default=0.15, help="Optional consistency loss weight on INT/EPI")

    ap.add_argument(
        "--stop_on",
        default="delta",
        choices=["delta", "mean", "avg_int_epi", "avg_mean_delta"],
        help="Validation metric for early stopping",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_prepared_split(str(Path(args.prepared_dir) / "train.tsv"))
    val_df = load_prepared_split(str(Path(args.prepared_dir) / "val.tsv"))
    test_df = load_prepared_split(str(Path(args.prepared_dir) / "test.tsv"))

    for col in ["mean", "delta", "log2_WT", "log2_MT"]:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column '{col}' in prepared TSV.")

    train_ds = PairedMeanDeltaDataset(train_df, rc_aug=args.rc_aug, rc_prob=args.rc_prob)
    val_ds = PairedMeanDeltaDataset(val_df, rc_aug=False)
    test_ds = PairedMeanDeltaDataset(test_df, rc_aug=False)

    device = torch.device("cpu")
    model = SmallCNNMeanDelta(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss == "smoothl1":
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.5)
    else:
        loss_fn = nn.MSELoss(reduction="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def _val_score(pack: Dict[str, np.ndarray]) -> Dict[str, float]:
        m_mean = compute_metrics(pack["y_mean"], pack["p_mean"])
        m_delta = compute_metrics(pack["y_delta"], pack["p_delta"])
        m_int = compute_metrics(pack["y_int"], pack["p_int"])
        m_epi = compute_metrics(pack["y_epi"], pack["p_epi"])
        return {
            "val_mean_pearson": float(m_mean["pearson"]),
            "val_mean_spearman": float(m_mean["spearman"]),
            "val_delta_pearson": float(m_delta["pearson"]),
            "val_delta_spearman": float(m_delta["spearman"]),
            "val_int_pearson": float(m_int["pearson"]),
            "val_int_spearman": float(m_int["spearman"]),
            "val_epi_pearson": float(m_epi["pearson"]),
            "val_epi_spearman": float(m_epi["spearman"]),
        }

    def _score(metrics_val: Dict[str, float]) -> float:
        if args.stop_on == "delta":
            return metrics_val["val_delta_pearson"]
        if args.stop_on == "mean":
            return metrics_val["val_mean_pearson"]
        if args.stop_on == "avg_int_epi":
            return 0.5 * (metrics_val["val_int_pearson"] + metrics_val["val_epi_pearson"])
        if args.stop_on == "avg_mean_delta":
            return 0.5 * (metrics_val["val_mean_pearson"] + metrics_val["val_delta_pearson"])
        return metrics_val["val_delta_pearson"]

    best_score = -1e9
    best_path = out_dir / "cnn_mean_delta_best.pt"
    bad = 0
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for (x, ym, yd, yi, ye, w_m, w_d, w_i, w_e) in train_loader:
            x = x.to(device)
            ym = ym.to(device)
            yd = yd.to(device)
            yi = yi.to(device)
            ye = ye.to(device)
            w_m = w_m.to(device)
            w_d = w_d.to(device)
            w_i = w_i.to(device)
            w_e = w_e.to(device)

            opt.zero_grad(set_to_none=True)
            pm, pd = model(x)

            # main losses
            l_mean = loss_fn(pm, ym) * w_m
            l_delta = loss_fn(pd, yd) * w_d
            loss = args.w_mean * l_mean + args.w_delta * l_delta

            # optional consistency
            if args.consistency > 0:
                pred_int = pm + 0.5 * pd
                pred_epi = pm - 0.5 * pd
                l_int = loss_fn(pred_int, yi) * w_i
                l_epi = loss_fn(pred_epi, ye) * w_e
                loss = loss + args.consistency * (l_int + l_epi)

            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * len(x)
            n += len(x)

        # validate
        val_pack = predict_all(model, val_loader, device)
        metrics_val = _val_score(val_pack)
        sc = _score(metrics_val)

        hist.append({"epoch": ep, "train_loss": total / max(1, n), "score": float(sc), **metrics_val})
        print(
            f"[epoch {ep:03d}] train_loss={total/max(1,n):.4f} score={sc:.4f} "
            f"val_delta={metrics_val['val_delta_pearson']:.4f} val_mean={metrics_val['val_mean_pearson']:.4f} "
            f"val_int={metrics_val['val_int_pearson']:.4f} val_epi={metrics_val['val_epi_pearson']:.4f}"
        )

        if sc > best_score + 1e-4:
            best_score = float(sc)
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: no improvement for {args.patience} epochs")
                break

    # test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_pack_best = predict_all(model, val_loader, device)

    test_pack = predict_all(model, test_loader, device)
    mt_mean = compute_metrics(test_pack["y_mean"], test_pack["p_mean"])
    mt_delta = compute_metrics(test_pack["y_delta"], test_pack["p_delta"])
    mt_int = compute_metrics(test_pack["y_int"], test_pack["p_int"])
    mt_epi = compute_metrics(test_pack["y_epi"], test_pack["p_epi"])

    metrics = {
        "model": "cnn_mean_delta",
        "seed": int(args.seed),
        "val_best_score": float(best_score),
        "test_mean_pearson": float(mt_mean["pearson"]),
        "test_mean_spearman": float(mt_mean["spearman"]),
        "test_delta_pearson": float(mt_delta["pearson"]),
        "test_delta_spearman": float(mt_delta["spearman"]),
        "test_int_pearson": float(mt_int["pearson"]),
        "test_int_spearman": float(mt_int["spearman"]),
        "test_epi_pearson": float(mt_epi["pearson"]),
        "test_epi_spearman": float(mt_epi["spearman"]),
        "n_test": int(len(test_pack["y_delta"]))
    }

    write_json(metrics, out_dir / "cnn_mean_delta.metrics.json")
    write_json({"history": hist}, out_dir / "cnn_mean_delta.history.json")

    # validation predictions (best checkpoint)
    vpred = val_df[["element_id"]].copy()
    vpred["y_int_log2WT"] = val_pack_best["y_int"]
    vpred["y_epi_log2MT"] = val_pack_best["y_epi"]
    vpred["y_mean"] = val_pack_best["y_mean"]
    vpred["y_delta"] = val_pack_best["y_delta"]
    vpred["pred_int"] = val_pack_best["p_int"]
    vpred["pred_epi"] = val_pack_best["p_epi"]
    vpred["pred_mean"] = val_pack_best["p_mean"]
    vpred["pred_delta"] = val_pack_best["p_delta"]
    vpred.to_csv(out_dir / "cnn_mean_delta.val_predictions.tsv", sep="\t", index=False)


    pred = test_df[["element_id"]].copy()
    pred["y_int_log2WT"] = test_pack["y_int"]
    pred["y_epi_log2MT"] = test_pack["y_epi"]
    pred["y_mean"] = test_pack["y_mean"]
    pred["y_delta"] = test_pack["y_delta"]
    pred["pred_int"] = test_pack["p_int"]
    pred["pred_epi"] = test_pack["p_epi"]
    pred["pred_mean"] = test_pack["p_mean"]
    pred["pred_delta"] = test_pack["p_delta"]
    pred.to_csv(out_dir / "cnn_mean_delta.test_predictions.tsv", sep="\t", index=False)

    print("DONE")
    print(metrics)


if __name__ == "__main__":
    main()
