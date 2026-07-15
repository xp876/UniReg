import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from common import (
    apply_trim,
    choose_trim_length,
    load_zip_splits,
    normalize_seq,
    read_splits_json,
    write_json,
    log2_safe,
)

def split_by_splits_json(all_df: pd.DataFrame, splits: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sp in ("train", "val", "test"):
        ids = set(map(str, splits.get(sp, [])))
        out[sp] = all_df[all_df["element_id"].astype(str).isin(ids)].copy()
    return out

def _add_sample_weight_per_element(df: pd.DataFrame) -> pd.DataFrame:
    if "element_id" not in df.columns:
        df["sample_weight"] = 1.0
        return df
    counts = df.groupby("element_id").size()
    df = df.copy()
    df["sample_weight"] = df["element_id"].map(lambda x: 1.0 / float(counts.loc[x]))
    return df

def _pivot_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form rows (context+replicate) into one row per element_id with MT_agg and WT_agg."""
    df = df.copy()
    # accept both 'sequence' and 'seq'
    if "seq" in df.columns and "sequence" not in df.columns:
        df = df.rename(columns={"seq": "sequence"})
    required = {"element_id", "sequence", "replicate", "activity_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; got {df.columns.tolist()}")
    # keep only agg measurements
    df = df[df["replicate"].isin(["MT_agg", "WT_agg"])].copy()
    # some exports include 'context'; we don't strictly need it here
    # sanity: each element should have exactly 2 rows
    # pivot values
    pv = df.pivot_table(index=["element_id", "sequence"], columns="replicate", values="activity_raw", aggfunc="first").reset_index()
    # ensure both present
    if "MT_agg" not in pv.columns or "WT_agg" not in pv.columns:
        raise ValueError(f"Pivot missing MT_agg/WT_agg columns; columns={pv.columns.tolist()}")
    # drop elements missing either side
    before = len(pv)
    pv = pv[pv["MT_agg"].notna() & pv["WT_agg"].notna()].copy()
    dropped = before - len(pv)
    pv["raw_MT"] = pv["MT_agg"].astype(float)
    pv["raw_WT"] = pv["WT_agg"].astype(float)
    # log2 transforms
    pv["log2_MT"] = pv["raw_MT"].map(log2_safe)
    pv["log2_WT"] = pv["raw_WT"].map(log2_safe)
    pv["delta"] = pv["log2_WT"] - pv["log2_MT"]          # log2(WT/MT)
    pv["mean"] = 0.5 * (pv["log2_WT"] + pv["log2_MT"])    # average log activity
    pv["len"] = pv["sequence"].astype(str).map(len).astype(int)
    pv["qc_flag"] = True
    pv.attrs["dropped_missing_pair"] = int(dropped)
    return pv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_zip", required=True, help="Export zip containing train/val/test.csv with both contexts (FormatB recommended).")
    ap.add_argument("--splits_json", required=True, help="Element-wise split json (train/val/test lists of element_id).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--auto_trim", default="yes", choices=["yes", "no"])
    ap.add_argument("--trim_to", type=int, default=0, help="Override trim length; 0 => choose mode length from train")
    ap.add_argument("--eps", type=float, default=1e-8, help="Clamp for log2 to avoid log(0)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_splits = load_zip_splits(args.data_zip)
    splits = read_splits_json(args.splits_json)

    all_df = pd.concat(in_splits.values(), ignore_index=True)
    # normalize sequences
    if "seq" in all_df.columns and "sequence" not in all_df.columns:
        all_df = all_df.rename(columns={"seq": "sequence"})
    all_df["sequence"] = all_df["sequence"].astype(str).map(normalize_seq)

    # rebuild train/val/test based on splits_json for consistency
    in_splits = split_by_splits_json(all_df, splits)

    out_splits = {}
    stats = {"dropped_missing_pair": {}}
    for sp, df in in_splits.items():
        pv = _pivot_pair(df)
        stats["dropped_missing_pair"][sp] = int(pv.attrs.get("dropped_missing_pair", 0))
        out_splits[sp] = pv

    # decide trim length
    trim_to = None
    if args.auto_trim == "yes":
        if args.trim_to > 0:
            trim_to = int(args.trim_to)
        else:
            trim_to = int(choose_trim_length(out_splits["train"]))
        for sp in out_splits:
            out_splits[sp] = apply_trim(out_splits[sp], trim_to)

    # add per-element sample_weight (mostly all 1.0 for FormatB; still good for generality)
    for sp in out_splits:
        out_splits[sp] = _add_sample_weight_per_element(out_splits[sp])

    # write splits
    for sp, df in out_splits.items():
        df.to_csv(out_dir / f"{sp}.tsv", sep="\t", index=False)

    qc = {
        "data_zip": str(args.data_zip),
        "splits_json": str(args.splits_json),
        "auto_trim": args.auto_trim,
        "trim_to": int(trim_to) if trim_to is not None else None,
        "n_rows": {sp: int(len(df)) for sp, df in out_splits.items()},
        "n_element_id": {sp: int(df["element_id"].nunique()) for sp, df in out_splits.items()},
        "dropped_missing_pair": stats["dropped_missing_pair"],
        "targets": ["log2_MT", "log2_WT", "delta", "mean"],
        "target_stats": {
            sp: {t: {
                "mean": float(pd.to_numeric(df[t], errors="coerce").mean()),
                "std": float(pd.to_numeric(df[t], errors="coerce").std(ddof=0)),
                "min": float(pd.to_numeric(df[t], errors="coerce").min()),
                "max": float(pd.to_numeric(df[t], errors="coerce").max()),
            } for t in ["log2_MT","log2_WT","delta","mean"]}
            for sp, df in out_splits.items()
        },
    }
    write_json(qc, out_dir / "qc_paired.json")
    print("Wrote paired dataset to:", out_dir)
    print(json.dumps(qc, indent=2))

if __name__ == "__main__":
    import json
    main()
