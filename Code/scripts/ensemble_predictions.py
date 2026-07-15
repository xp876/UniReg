
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from common import compute_metrics, write_json


PRED_ALIASES = {
    "pred_int": ["pred_wt", "pred_WT", "pred_log2_WT", "pred_Wt"],
    "pred_epi": ["pred_mt", "pred_MT", "pred_log2_MT", "pred_Mt"],
    "pred_delta": ["pred_delta", "pred_Delta", "pred_d"],
}

Y_ALIASES = {
    # NOTE: Several training scripts historically wrote WT/MT labels as y_wt/y_mt.
    # Keep robust aliasing here so ensembling doesn't fail across versions.
    "y_int": ["y_int", "log2_WT", "y_log2_WT", "WT", "y_wt", "y_WT"],
    "y_epi": ["y_epi", "log2_MT", "y_log2_MT", "MT", "y_mt", "y_MT"],
    "y_delta": ["y_delta", "delta", "y_d"],
}


def _ensure_col(df: pd.DataFrame, want: str, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """Ensure column `want` exists by copying from the first available alias."""
    if want in df.columns:
        return df
    for a in aliases.get(want, []):
        if a in df.columns:
            df[want] = df[a]
            return df
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_tsvs", required=True, help="Comma-separated list of prediction TSVs (same split/test set).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="ensemble")
    ap.add_argument("--out_pred_tsv", default="", help="Optional explicit path for the output prediction TSV (enables ensembling val splits).")
    ap.add_argument("--y_cols", default="y_delta", help="Comma-separated y columns")
    ap.add_argument("--pred_cols", default="pred_delta", help="Comma-separated pred columns matching y_cols")
    ap.add_argument("--id_col", default="element_id")
    args = ap.parse_args()

    pred_paths = [Path(p.strip()) for p in args.pred_tsvs.split(",") if p.strip()]
    if len(pred_paths) < 2:
        raise SystemExit("Need at least 2 prediction TSVs for ensemble.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(p, sep="\t") for p in pred_paths]
    base = dfs[0].copy()

    y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    pred_cols = [c.strip() for c in args.pred_cols.split(",") if c.strip()]
    if len(y_cols) != len(pred_cols):
        raise SystemExit("y_cols and pred_cols must have same length")

    # align by id_col (keep order of base)
    if args.id_col not in base.columns:
        raise SystemExit(f"Missing id_col={args.id_col} in predictions")

    # be backward-compatible with older column naming
    for yc in y_cols:
        base = _ensure_col(base, yc, Y_ALIASES)
        if yc not in base.columns:
            raise SystemExit(f"Missing y col {yc} (and no alias found) in base TSV")

    # set base index for stable alignment
    base_ids = base[args.id_col].astype(str).values

    # merge predicted columns across dfs (robust to column aliases + row order)
    for y_col, p_col in zip(y_cols, pred_cols):
        mats = []
        kept = 0
        for i, d in enumerate(dfs):
            if args.id_col not in d.columns:
                print(f"WARN: skip TSV missing id_col={args.id_col}: {pred_paths[i]}")
                continue
            d = d.copy()
            d[args.id_col] = d[args.id_col].astype(str)
            d = _ensure_col(d, p_col, PRED_ALIASES)
            if p_col not in d.columns:
                print(f"WARN: skip TSV missing pred col {p_col} (no alias found): {pred_paths[i]}")
                continue
            s = pd.to_numeric(d.set_index(args.id_col)[p_col], errors="coerce")
            mats.append(s.reindex(base_ids).to_numpy(dtype=float))
            kept += 1
        if kept < 1:
            raise SystemExit(f"No usable TSVs for pred col {p_col}")
        P = np.vstack(mats)
        base[p_col] = np.nanmean(P, axis=0)

    # write ensemble predictions
    out_pred = Path(args.out_pred_tsv) if args.out_pred_tsv else (out_dir / f"{args.model_name}.test_predictions.tsv")
    base.to_csv(out_pred, sep="\t", index=False)

    # compute metrics (for first y/p pair, plus extras)
    metrics: Dict[str, float] = {"model": args.model_name}
    for y_col, p_col in zip(y_cols, pred_cols):
        y = pd.to_numeric(base[y_col], errors="coerce").values.astype(float)
        p = pd.to_numeric(base[p_col], errors="coerce").values.astype(float)
        m = np.isfinite(y) & np.isfinite(p)
        met = compute_metrics(y[m], p[m])
        for k, v in met.items():
            metrics[f"test_{y_col.replace('y_','')}_{k}"] = float(v)
        metrics[f"n_{y_col.replace('y_','')}"] = int(m.sum())

    out_metrics = out_dir / f"{args.model_name}.metrics.json"
    write_json(metrics, str(out_metrics))
    print("Wrote:", out_pred)
    print("Wrote:", out_metrics)
    print(metrics)

if __name__ == "__main__":
    main()
