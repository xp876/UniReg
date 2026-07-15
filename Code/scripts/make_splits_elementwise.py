import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from common import load_zip_splits, normalize_seq, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_zip", required=True, help="FormatB_agg_only.zip")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--require_paired", action="store_true", help="Only keep element_ids that have both MT_agg and WT_agg")
    args = ap.parse_args()

    df = pd.concat(load_zip_splits(args.data_zip).values(), ignore_index=True)
    if "seq" in df.columns and "sequence" not in df.columns:
        df = df.rename(columns={"seq": "sequence"})
    df["element_id"] = df["element_id"].astype(str)

    if args.require_paired and "replicate" in df.columns:
        # keep those that have both contexts measured
        sub = df[df["replicate"].isin(["MT_agg", "WT_agg"])].copy()
        counts = sub.groupby(["element_id", "replicate"]).size().unstack(fill_value=0)
        keep = counts[(counts.get("MT_agg", 0) > 0) & (counts.get("WT_agg", 0) > 0)].index.astype(str)
        ids = np.array(sorted(set(keep)))
    else:
        ids = np.array(sorted(df["element_id"].unique().tolist()))

    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * float(args.train_frac)))
    n_val = int(round(n * float(args.val_frac)))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train : n_train + n_val].tolist()
    test_ids = ids[n_train + n_val :].tolist()

    out = {"train": train_ids, "val": val_ids, "test": test_ids}
    write_json(out, args.out_json)

    print("Wrote splits to:", args.out_json)
    print({"n": n, "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)})


if __name__ == "__main__":
    main()
