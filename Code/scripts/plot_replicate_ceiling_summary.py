import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description='Plot replicate ceiling summaries (bar plots for DELTA ceilings).')
    ap.add_argument('--ceiling_tsv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--context', default='DELTA', choices=['INT','EPI','DELTA'])
    ap.add_argument('--metric', default='pairwise_mean_pearson')
    ap.add_argument('--prefix', default='R', help='Element prefix to plot (e.g., R, A, C, ALL)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ceiling_tsv, sep='\t')
    sub = df[(df['context'] == args.context) & (df['metric'] == args.metric) & (df['prefix'].astype(str) == args.prefix)].copy()
    if sub.empty:
        # fallback to ALL
        sub = df[(df['context'] == args.context) & (df['metric'] == args.metric) & (df['prefix'].astype(str) == 'ALL')].copy()

    sub = sub.dropna(subset=['value']).copy()
    if sub.empty:
        print('Nothing to plot')
        return

    sub = sub.sort_values('value', ascending=False)

    plt.figure(figsize=(max(6, 0.45*len(sub)), 3.2))
    plt.bar(sub['label'].astype(str).tolist(), sub['value'].astype(float).tolist())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f"{args.context} ceiling ({args.metric})")
    plt.title(f"Replicate ceiling by run (prefix={args.prefix})")
    plt.tight_layout()
    out_png = out_dir / f"ceiling_{args.context}_{args.metric}_prefix{args.prefix}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print('Wrote:', out_png)


if __name__ == '__main__':
    main()
