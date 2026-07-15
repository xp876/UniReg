import argparse
from pathlib import Path
from typing import Tuple

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

TOK = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

def encode_seq(seq: str) -> np.ndarray:
    return np.array([TOK.get(ch, 4) for ch in seq], dtype=np.int64)

def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(c, "N") for c in seq[::-1])

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: str, rc_aug: bool = False, rc_prob: float = 0.5):
        self.seqs = df["sequence"].astype(str).tolist()
        self.y = pd.to_numeric(df[target], errors="coerce").values.astype(np.float32)
        weight_col = None
        if target == "delta" and "w_delta" in df.columns:
            weight_col = "w_delta"
        elif target == "mean" and "w_mean" in df.columns:
            weight_col = "w_mean"
        elif target == "log2_WT" and "w_int" in df.columns:
            weight_col = "w_int"
        elif target == "log2_MT" and "w_epi" in df.columns:
            weight_col = "w_epi"

        if weight_col is not None:
            self.w = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).values.astype(np.float32)
        else:
            self.w = pd.to_numeric(df.get("sample_weight", pd.Series([1.0] * len(df))), errors="coerce").fillna(1.0).values.astype(np.float32)
        self.rc_aug = rc_aug
        self.rc_prob = float(rc_prob)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        if self.rc_aug and (np.random.rand() < self.rc_prob):
            s = reverse_complement(s)
        x = encode_seq(s)
        return torch.from_numpy(x), torch.tensor(self.y[idx]), torch.tensor(self.w[idx])

class SmallCNN(nn.Module):
    def __init__(self, vocab_size: int = 5, d: int = 32, dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x).transpose(1, 2)  # (B, d, L)
        h1 = self.act(self.conv1(h))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h1))
        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h2, dim=-1)
        p3 = torch.amax(h3, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        out = self.head(feat).squeeze(-1)
        return out

def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            ps.append(p)
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", required=True, help="Target column name (log2_WT, log2_MT, delta, mean).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--rc_aug", action="store_true")
    ap.add_argument("--rc_prob", type=float, default=0.5)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_prepared_split(str(Path(args.prepared_dir) / "train.tsv"))
    val_df = load_prepared_split(str(Path(args.prepared_dir) / "val.tsv"))
    test_df = load_prepared_split(str(Path(args.prepared_dir) / "test.tsv"))

    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in prepared TSV. Columns: {train_df.columns.tolist()}")

    train_ds = SeqDataset(train_df, args.target, rc_aug=args.rc_aug, rc_prob=args.rc_prob)
    val_ds = SeqDataset(val_df, args.target, rc_aug=False)
    test_ds = SeqDataset(test_df, args.target, rc_aug=False)

    device = torch.device("cpu")
    model = SmallCNN(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.5)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_val = -1e9
    best_path = out_dir / "cnn_best.pt"
    bad = 0
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss = (loss * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(y)
            n += len(y)

        yv, pv = eval_model(model, val_loader, device)
        m = compute_metrics(yv, pv)
        val_p = m["pearson"]
        hist.append({"epoch": ep, "train_loss": total / max(1, n), "val_pearson": val_p, "val_spearman": m["spearman"]})
        print(f"[epoch {ep:03d}] train_loss={total/max(1,n):.4f} val_pearson={val_p:.4f}")

        if val_p > best_val + 1e-4:
            best_val = val_p
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: no improvement for {args.patience} epochs")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # also compute validation predictions (best checkpoint)
    yv_best, pv_best = eval_model(model, val_loader, device)

    yt, pt = eval_model(model, test_loader, device)
    mt = compute_metrics(yt, pt)
    metrics = {
        "model": "cnn_single",
        "target": args.target,
        "seed": int(args.seed),
        "val_best_pearson": float(best_val),
        "test_pearson": float(mt["pearson"]),
        "test_spearman": float(mt["spearman"]),
        "n_test": int(len(yt)),
    }

    # write metrics (keep legacy filenames + add model-tagged filenames)
    tag = out_dir.name  # e.g. cnn_delta, cnn_mean, etc.
    write_json(metrics, out_dir / "cnn.metrics.json")
    write_json(metrics, out_dir / f"{tag}.metrics.json")
    write_json({"history": hist}, out_dir / "cnn.history.json")
    write_json({"history": hist}, out_dir / f"{tag}.history.json")
    # write predictions with consistent column names
    # map target -> standardized y/p column names
    y_col_map = {
        'delta': ('y_delta', 'pred_delta', 'delta'),
        'mean': ('y_mean', 'pred_mean', 'mean'),
        'log2_WT': ('y_int', 'pred_int', 'log2_WT'),
        'log2_MT': ('y_epi', 'pred_epi', 'log2_MT'),
    }
    if args.target not in y_col_map:
        raise ValueError(f'Unsupported target for standardized outputs: {args.target}')

    y_col, p_col, src_y = y_col_map[args.target]

    # validation predictions table (for stacking / fusion)
    val_pred = val_df[['element_id']].copy()
    if 'sequence' in val_df.columns:
        val_pred['sequence'] = val_df['sequence'].values
    for col in ['log2_WT','log2_MT','delta','mean','raw_WT','raw_MT']:
        if col in val_df.columns and col not in val_pred.columns:
            val_pred[col] = val_df[col].values
    val_pred[y_col] = val_df[src_y].values
    val_pred[p_col] = pv_best
    val_pred.to_csv(out_dir / 'cnn.val_predictions.tsv', sep='	', index=False)
    val_pred.to_csv(out_dir / f'{tag}.val_predictions.tsv', sep='	', index=False)


    pred = test_df[['element_id']].copy()
    if 'sequence' in test_df.columns:
        pred['sequence'] = test_df['sequence'].values

    # keep raw fields if present (helpful for diagnostics)
    for col in ['log2_WT','log2_MT','delta','mean','raw_WT','raw_MT']:
        if col in test_df.columns and col not in pred.columns:
            pred[col] = test_df[col].values

    # standardized y + pred columns
    pred[y_col] = test_df[src_y].values
    pred[p_col] = pt

    # legacy file + model-tagged file
    pred.to_csv(out_dir / 'cnn.test_predictions.tsv', sep='	', index=False)
    pred.to_csv(out_dir / f'{tag}.test_predictions.tsv', sep='	', index=False)

    print("DONE")
    print(metrics)

if __name__ == "__main__":
    main()
