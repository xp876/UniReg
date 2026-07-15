import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_placeholder(out_png: Path, title: str, msg: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.38, msg, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Wrote placeholder:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer_tsv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--metric", default="test_pearson", help="Column in transfer TSV (e.g., test_pearson, test_pearson_R)")
    ap.add_argument("--title", default="Transfer matrix")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    out_png = Path(args.out_png)
    df = pd.read_csv(args.transfer_tsv, sep="\t")
    if df.empty:
        return _save_placeholder(out_png, args.title, f"No rows in {Path(args.transfer_tsv).name}")

    if args.metric not in df.columns:
        cols = ", ".join(df.columns.tolist())
        return _save_placeholder(out_png, args.title, f"Metric '{args.metric}' not found.\nColumns: {cols[:180]}")

    # Keep only rows with valid identifiers and numeric metric values
    df = df.copy()
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    for c in ["source", "target"]:
        if c not in df.columns:
            return _save_placeholder(out_png, args.title, f"Missing required column: {c}")
    df = df.dropna(subset=["source", "target", args.metric])

    if df.empty:
        return _save_placeholder(out_png, args.title, f"No finite values for metric '{args.metric}'")

    # average across seeds
    df2 = df.groupby(["source", "target"], as_index=False)[args.metric].mean()
    if df2.empty:
        return _save_placeholder(out_png, args.title, "No source-target pairs after aggregation")

    sources = sorted(df2["source"].astype(str).unique().tolist())
    targets = sorted(df2["target"].astype(str).unique().tolist())

    if len(sources) == 0 or len(targets) == 0:
        return _save_placeholder(out_png, args.title, "No sources/targets to plot")

    mat = np.full((len(sources), len(targets)), np.nan, dtype=float)
    src_idx = {s: i for i, s in enumerate(sources)}
    tgt_idx = {t: j for j, t in enumerate(targets)}
    for _, r in df2.iterrows():
        i = src_idx.get(str(r["source"]))
        j = tgt_idx.get(str(r["target"]))
        if i is not None and j is not None:
            mat[i, j] = float(r[args.metric])

    # robust figure size; avoid 0-dimension layout edge-cases
    w = max(5.0, 1.05 * len(targets) + 2.8)
    h = max(3.5, 0.72 * len(sources) + 2.3)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=args.vmin, vmax=args.vmax)
    ax.set_xticks(range(len(targets)))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticklabels(sources)
    ax.set_title(f"{args.title}\n(metric={args.metric})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric)

    # annotate cells
    for i in range(len(sources)):
        for j in range(len(targets)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Constrained layout can crash on some matplotlib versions for edge-case grids; use tight_layout safely.
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"WARN: tight_layout failed ({e}); saving without layout adjustment")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()
