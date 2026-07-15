"""Reviewer-proof robustness check: drop near-duplicates / highly similar test elements.

Goal:
- For each split_seed, compute max cosine similarity (TF-IDF k-mer) between each test element and all train elements.
- Drop the top q fraction (default 1%) of test elements by similarity.
- Recompute Pearson correlation on remaining elements.
- Aggregate across split seeds (mean) and bootstrap CI.

This is cheap, does not require re-training, and helps rebut leakage/near-duplicate concerns.

Inputs:
- --root: the plan8 output directory (contains prepared/ and results/)
- --model_pattern: glob for per-split prediction TSVs, e.g. "cnn_wt_mt_delta3head_ens.test_predictions.tsv"

Outputs:
- similarity_robustness.tsv
- similarity_robustness.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 8:
        return float("nan")
    return float(np.corrcoef(y[m], p[m])[0, 1])


def load_preds(pred_tsv: Path, y_col: str = "y_delta", p_col: str = "pred_delta") -> pd.DataFrame:
    df = pd.read_csv(pred_tsv, sep="\t")
    if "element_id" not in df.columns:
        raise SystemExit(f"Missing element_id in {pred_tsv}")
    if y_col not in df.columns or p_col not in df.columns:
        raise SystemExit(f"Missing {y_col}/{p_col} in {pred_tsv}")
    out = df[["element_id", y_col, p_col]].copy()
    out["prefix"] = out["element_id"].astype(str).map(element_prefix)
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out = out.dropna(subset=[y_col, p_col]).copy()
    out = out.rename(columns={y_col: "y", p_col: "p"})
    return out


def subset(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "overall":
        return df
    if mode == "non_control":
        return df[df["prefix"] != "C"].copy()
    if mode in {"R", "A", "C"}:
        return df[df["prefix"] == mode].copy()
    raise ValueError(mode)


def compute_max_sim(train_seqs: List[str], test_seqs: List[str], k: int = 6) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    Xtr = vec.fit_transform(train_seqs)
    Xte = vec.transform(test_seqs)
    # cosine similarity with l2-normalized tfidf == dot product
    sims = Xte @ Xtr.T
    # max per test row
    # sparse matrix -> efficient max
    max_sim = np.asarray(sims.max(axis=1)).ravel().astype(float)
    return max_sim


def boot_mean(vals_by_split: Dict[str, np.ndarray], n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    splits = sorted(vals_by_split.keys())
    point = float(np.nanmean([np.nanmean(vals_by_split[s]) for s in splits]))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # resample splits with replacement
        chosen = rng.choice(splits, size=len(splits), replace=True)
        boots[i] = float(np.nanmean([np.nanmean(vals_by_split[s]) for s in chosen]))
    lo, hi = float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))
    return point, (lo, hi)


def collect_split_paths(root: Path, model_pattern: str) -> Dict[str, Path]:
    out = {}
    for p in root.rglob(model_pattern):
        m = re.search(r"split_seed\d+", str(p))
        if m:
            out[m.group(0)] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model_pattern", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--q", type=float, default=0.99, help="Quantile threshold to DROP (default 0.99 = drop top 1%)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = collect_split_paths(root, args.model_pattern)
    if not pred_paths:
        raise SystemExit(f"No preds found for pattern={args.model_pattern} under {root}")

    strata = ["overall", "non_control", "R"]
    rows = []
    report = {
        "root": str(root),
        "model_pattern": args.model_pattern,
        "q": float(args.q),
        "k": int(args.k),
        "splits": sorted(pred_paths.keys()),
        "by_stratum": {},
    }

    for st in strata:
        per_split_orig: Dict[str, np.ndarray] = {}
        per_split_filt: Dict[str, np.ndarray] = {}

        for sp, pred_tsv in pred_paths.items():
            prep_dir = root / "prepared" / sp / "weighted"
            train = pd.read_csv(prep_dir / "train.tsv", sep="\t")
            test = pd.read_csv(prep_dir / "test.tsv", sep="\t")

            # align sequences and element_ids
            train_seq = train["sequence"].fillna("").astype(str).tolist()
            test_df = test[["element_id", "sequence"]].copy()
            test_df["sequence"] = test_df["sequence"].fillna("").astype(str)

            max_sim = compute_max_sim(train_seq, test_df["sequence"].tolist(), k=args.k)
            sim_df = pd.DataFrame({"element_id": test_df["element_id"].astype(str).values, "max_sim": max_sim})

            pred = load_preds(pred_tsv)
            pred = subset(pred, st)
            m = pred.merge(sim_df, on="element_id", how="inner")
            if m.empty:
                continue

            y = m["y"].values.astype(float)
            p = m["p"].values.astype(float)
            per_split_orig[sp] = np.array([pearson(y, p)], dtype=float)

            thr = float(np.nanquantile(m["max_sim"].values.astype(float), args.q))
            keep = m["max_sim"].values.astype(float) <= thr
            y2, p2 = y[keep], p[keep]
            per_split_filt[sp] = np.array([pearson(y2, p2)], dtype=float)

            rows.append({
                "split": sp,
                "stratum": st,
                "pearson_orig": float(per_split_orig[sp][0]),
                "pearson_drop_top": float(per_split_filt[sp][0]),
                "drop_q": float(args.q),
                "threshold": thr,
                "n_before": int(len(y)),
                "n_after": int(len(y2)),
            })

        # aggregate
        if per_split_orig and per_split_filt:
            orig_vals = {k: v for k, v in per_split_orig.items()}
            filt_vals = {k: v for k, v in per_split_filt.items()}
            orig_point, orig_ci = boot_mean(orig_vals, args.n_boot, args.seed + hash((args.model_pattern, st, "orig")) % 100000)
            filt_point, filt_ci = boot_mean(filt_vals, args.n_boot, args.seed + hash((args.model_pattern, st, "filt")) % 100000)
            report["by_stratum"][st] = {
                "orig": {"point": orig_point, "ci95": orig_ci},
                "drop_top": {"point": filt_point, "ci95": filt_ci},
            }

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "similarity_robustness.tsv", sep="\t", index=False)
    with open(out_dir / "similarity_robustness.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
