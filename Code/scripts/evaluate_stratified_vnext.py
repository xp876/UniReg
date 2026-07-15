
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from common import compute_metrics, write_json


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def _safe_roc_auc(y01: np.ndarray, score: np.ndarray) -> float:
    y01 = y01.astype(int)
    if np.unique(y01).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y01, score))
    except Exception:
        return float("nan")


def _safe_auprc(y01: np.ndarray, score: np.ndarray) -> float:
    y01 = y01.astype(int)
    if np.unique(y01).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y01, score))
    except Exception:
        return float("nan")


def metrics_for(df: pd.DataFrame, y_col: str, pred_col: str) -> Dict[str, float]:
    y = pd.to_numeric(df[y_col], errors="coerce").values.astype(float)
    p = pd.to_numeric(df[pred_col], errors="coerce").values.astype(float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 5:
        return {"n": int(m.sum())}
    met = compute_metrics(y[m], p[m])
    # sign classification diagnostics (delta-like)
    y01 = (y[m] > 0).astype(int)
    p01 = (p[m] > 0).astype(int)
    out = {
        "n": int(m.sum()),
        **{k: float(v) for k, v in met.items()},
        "acc_sign": float(accuracy_score(y01, p01)) if m.sum() > 0 else float("nan"),
        "auc_sign": _safe_roc_auc(y01, p[m]),
        "auprc_sign": _safe_auprc(y01, p[m]),
    }
    return out


def make_bins_abs(abs_y: np.ndarray, abs_bins: List[float]) -> List[np.ndarray]:
    masks = []
    for a, b in zip(abs_bins[:-1], abs_bins[1:]):
        masks.append((abs_y >= a) & (abs_y < b))
    return masks


def make_bins_quantile(abs_y: np.ndarray, n_bins: int) -> List[np.ndarray]:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(abs_y[np.isfinite(abs_y)], qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    masks = []
    for a, b in zip(edges[:-1], edges[1:]):
        masks.append((abs_y >= a) & (abs_y < b))
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_tsv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--y_col", default="y_delta")
    ap.add_argument("--pred_col", default="pred_delta")
    ap.add_argument("--exclude_prefixes", default="", help="Comma-separated prefixes to exclude (e.g., C)")
    ap.add_argument("--bins_mode", default="quantile", choices=["quantile", "abs"])
    ap.add_argument("--n_bins", type=int, default=6)
    ap.add_argument("--abs_bins", default="0,0.5,1.0,1.5,2.0,999")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_tsv, sep="\t")
    if "element_id" in df.columns:
        df["prefix"] = df["element_id"].astype(str).map(element_prefix)
    else:
        df["prefix"] = "UNK"

    if args.y_col not in df.columns or args.pred_col not in df.columns:
        raise SystemExit(f"Missing {args.y_col} or {args.pred_col} in {args.pred_tsv}")

    excl = [x.strip() for x in args.exclude_prefixes.split(",") if x.strip()]
    df_non = df[~df["prefix"].isin(excl)].copy() if excl else df.copy()

    out = {
        "pred_tsv": str(args.pred_tsv),
        "y_col": args.y_col,
        "pred_col": args.pred_col,
        "exclude_prefixes": excl,
        "overall": metrics_for(df, args.y_col, args.pred_col),
        "overall_non_excluded": metrics_for(df_non, args.y_col, args.pred_col),
        "by_prefix": {},
        "by_bin": [],
    }

    for pref, sub in df.groupby("prefix"):
        out["by_prefix"][pref] = metrics_for(sub, args.y_col, args.pred_col)

    # bins by |y|
    abs_y = np.abs(pd.to_numeric(df[args.y_col], errors="coerce").values.astype(float))
    if args.bins_mode == "abs":
        bins = [float(x) for x in args.abs_bins.split(",") if x.strip()]
        masks = make_bins_abs(abs_y, bins)
        labels = [f"[{a},{b})" for a, b in zip(bins[:-1], bins[1:])]
    else:
        masks = make_bins_quantile(abs_y, args.n_bins)
        # label with quantile index
        labels = [f"q{i+1}/{args.n_bins}" for i in range(len(masks))]

    for lab, m in zip(labels, masks):
        sub = df[m].copy()
        if len(sub) < 8:
            continue
        met = metrics_for(sub, args.y_col, args.pred_col)
        out["by_bin"].append({"bin": lab, **met})

    write_json(out, args.out_json)
    print("Wrote:", args.out_json)
    print("Overall:", out["overall"])

if __name__ == "__main__":
    main()
