#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--std_tsv', required=True)
    ap.add_argument('--splits_json', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.std_tsv, sep='\t')
    with open(args.splits_json, 'r', encoding='utf-8') as f:
        sp = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train','val','test']:
        keep = set(map(str, sp[split]))
        sub = df[df['element_id'].astype(str).isin(keep)].copy()

        # compatibility with existing load_prepared_split()/paired transfer code
        if 'sequence' not in sub.columns:
            if 'sequence_ref' in sub.columns:
                sub['sequence'] = sub['sequence_ref']
            elif 'sequence_alt' in sub.columns:
                sub['sequence'] = sub['sequence_alt']

        if 'sample_weight' not in sub.columns:
            sub['sample_weight'] = 1.0

        sub.to_csv(out_dir / f'{split}.tsv', sep='\t', index=False)

    print(f'[done] wrote prepared dir: {out_dir}')


if __name__ == '__main__':
    main()
