#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--std_tsv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--source_prepared_glob', required=True, help='e.g. /proj/out_plan8/prepared/split_seed{seed}/weighted')
    ap.add_argument('--split_seeds', default='0,1,2,3,4')
    ap.add_argument('--model_seeds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=15)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ext_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_seeds = [int(x) for x in args.split_seeds.split(',') if x.strip()]
    model_seeds = [int(x) for x in args.model_seeds.split(',') if x.strip()]

    rows = []
    for s in split_seeds:
        target_dir = out_dir / f'target_prepared_split_seed{s}'
        splits_json = out_dir / f'variant_split_seed{s}.json'
        subprocess.run(['python', str(ext_dir / 'make_simple_elementwise_splits.py'), '--in_tsv', args.std_tsv, '--seed', str(s), '--out_json', str(splits_json)], check=True)
        subprocess.run(['python', str(ext_dir / 'build_variant_prepared_dir.py'), '--std_tsv', args.std_tsv, '--splits_json', str(splits_json), '--out_dir', str(target_dir)], check=True)
        source_prepared = Path(args.source_prepared_glob.format(seed=s))
        if not source_prepared.exists():
            raise SystemExit(f'Missing source prepared dir: {source_prepared}')
        for m in model_seeds:
            run_dir = out_dir / 'transfer_runs' / f'source_split{s}' / f'model_seed{m}'
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                'python', str(ext_dir / 'transfer_train_eval_paired.py'),
                '--source_prepared', str(source_prepared),
                '--target_prepared', str(target_dir),
                '--out_dir', str(run_dir),
                '--seed', str(m),
                '--epochs', str(args.epochs),
                '--patience', str(args.patience),
                '--rc_aug',
            ]
            subprocess.run(cmd, check=True)
            with open(run_dir / 'transfer_paired.metrics.json', 'r', encoding='utf-8') as f:
                met = json.load(f)
            met['source_split_seed'] = s
            met['model_seed'] = m
            rows.append(met)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'variant_transfer_summary.tsv', sep='\t', index=False)
    print(f'[done] wrote transfer summary -> {out_dir / "variant_transfer_summary.tsv"}')


if __name__ == '__main__':
    main()
