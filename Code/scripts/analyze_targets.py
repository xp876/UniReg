
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import write_json


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prepared_dir / f"{args.split}.tsv", sep="\t")
    df["prefix"] = df["element_id"].astype(str).map(element_prefix)

    # basic stats
    stats = {
        "split": args.split,
        "n": int(len(df)),
        "n_prefix": df["prefix"].value_counts().to_dict(),
        "delta_quantiles": df["delta"].quantile([0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1]).to_dict(),
        "mean_quantiles": df["mean"].quantile([0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1]).to_dict(),
        "wt_mt_corr": float(df[["log2_WT","log2_MT"]].corr(method="pearson").iloc[0,1]),
        "delta_mean_corr": float(df[["delta","mean"]].corr(method="pearson").iloc[0,1]),
    }
    write_json(stats, str(out_dir / f"stats_{args.split}{('_'+args.tag) if args.tag else ''}.json"))

    # scatter WT vs MT
    plt.figure(figsize=(4,4))
    plt.scatter(df["log2_MT"], df["log2_WT"], s=6)
    plt.xlabel("log2_MT (EPI)")
    plt.ylabel("log2_WT (INT)")
    plt.title(f"{args.split}: WT vs MT")
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_wt_vs_mt_{args.split}{('_'+args.tag) if args.tag else ''}.png", dpi=200)
    plt.close()

    # hist delta
    plt.figure(figsize=(5,3))
    plt.hist(df["delta"].values, bins=60)
    plt.xlabel("delta = log2(WT/MT)")
    plt.ylabel("count")
    plt.title(f"{args.split}: delta distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_delta_{args.split}{('_'+args.tag) if args.tag else ''}.png", dpi=200)
    plt.close()

    # delta vs mean scatter
    plt.figure(figsize=(4,4))
    plt.scatter(df["mean"], df["delta"], s=6)
    plt.xlabel("mean")
    plt.ylabel("delta")
    plt.title(f"{args.split}: delta vs mean")
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_delta_vs_mean_{args.split}{('_'+args.tag) if args.tag else ''}.png", dpi=200)
    plt.close()

    print("Wrote analysis to", out_dir)

if __name__ == "__main__":
    main()
