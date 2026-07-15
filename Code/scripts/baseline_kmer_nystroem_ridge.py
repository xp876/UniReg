"""Kernel-method baseline: TF-IDF k-mers -> Nystroem (RBF) -> Ridge regression.

This baseline is primarily for reviewer-proof coverage of the "kernel methods" family
without heavy tuning.

Prepared TSV expectations:
  - element_id
  - sequence (or seq)
  - regression target column (usually: delta)

Standardized outputs (for target=delta):
  - <tag>.val_predictions.tsv  (columns include: element_id, y_delta, pred_delta)
  - <tag>.test_predictions.tsv (columns include: element_id, y_delta, pred_delta)
  - <tag>.meta.json

The loader is robust to legacy naming where the prepared TSV already contains y_delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import load_prepared_split, seed_everything


def _pick_src_y(df: pd.DataFrame, target: str) -> str:
    """Find the source y column in prepared TSV."""
    if target in df.columns:
        return target
    if target == "delta":
        for alt in ["y_delta", "log2ratio", "log2_ratio", "delta_log2", "y"]:
            if alt in df.columns:
                return alt
    raise SystemExit(
        f"Missing target '{target}' in prepared TSV. Available columns={df.columns.tolist()}"
    )


def _pick_weight(df: pd.DataFrame, target: str) -> np.ndarray:
    """Pick an optional per-sample weight column."""
    if target == "delta" and "w_delta" in df.columns:
        return pd.to_numeric(df["w_delta"], errors="coerce").fillna(1.0).values.astype(float)
    if "sample_weight" in df.columns:
        return pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(float)
    return np.ones(len(df), dtype=float)


def _load(prepared_dir: Path, split: str, target: str) -> pd.DataFrame:
    df = load_prepared_split(str(prepared_dir / f"{split}.tsv"))
    if "element_id" not in df.columns:
        if "eid" in df.columns:
            df = df.rename(columns={"eid": "element_id"})
        else:
            raise SystemExit(f"Missing element_id in {prepared_dir}/{split}.tsv")

    src_y = _pick_src_y(df, target)
    out = df[["element_id", "sequence", src_y]].copy().rename(columns={src_y: "_y"})
    # attach weights *before* dropping NaNs so alignment is preserved
    out["_w"] = _pick_weight(df, target)
    out["sequence"] = out["sequence"].fillna("").astype(str)
    out["_y"] = pd.to_numeric(out["_y"], errors="coerce")
    out = out.dropna(subset=["_y"]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", default="delta", choices=["delta"], help="Prepared TSV target column (usually: delta)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_components", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=1e-3, help="RBF gamma for Nystroem")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    args = ap.parse_args()

    seed_everything(args.seed)

    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "kmer_nystroem_ridge_delta"

    tr = _load(prepared_dir, "train", args.target)
    va = _load(prepared_dir, "val", args.target)
    te = _load(prepared_dir, "test", args.target)

    # Pipeline: TF-IDF k-mers -> Nystroem RBF -> standardize -> ridge
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(args.k, args.k), lowercase=False)),
        ("nystroem", Nystroem(kernel="rbf", gamma=args.gamma, n_components=args.n_components, random_state=args.seed)),
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=args.alpha, random_state=args.seed)),
    ])

    Xtr = tr["sequence"].tolist()
    ytr = tr["_y"].values.astype(float)
    wtr = tr["_w"].values.astype(float)
    pipe.fit(Xtr, ytr, ridge__sample_weight=wtr)

    def _pred(df: pd.DataFrame) -> pd.DataFrame:
        p = pipe.predict(df["sequence"].tolist()).astype(float)
        return pd.DataFrame({
            "element_id": df["element_id"].astype(str).values,
            "y_delta": df["_y"].values.astype(float),
            "pred_delta": p,
        })

    val_pred = _pred(va)
    test_pred = _pred(te)

    val_pred.to_csv(out_dir / f"{tag}.val_predictions.tsv", sep="\t", index=False)
    test_pred.to_csv(out_dir / f"{tag}.test_predictions.tsv", sep="\t", index=False)

    meta = {
        "model": tag,
        "prepared_dir": str(prepared_dir),
        "target": args.target,
        "k": int(args.k),
        "n_components": int(args.n_components),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
    }
    with open(out_dir / f"{tag}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
