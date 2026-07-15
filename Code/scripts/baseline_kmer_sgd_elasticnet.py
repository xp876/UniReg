import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline

from common import compute_metrics, load_prepared_split, seed_everything, write_json


def pick_weight_col(df: pd.DataFrame, target: str) -> str | None:
    if target == 'delta' and 'w_delta' in df.columns:
        return 'w_delta'
    if target == 'mean' and 'w_mean' in df.columns:
        return 'w_mean'
    if target == 'log2_WT' and 'w_int' in df.columns:
        return 'w_int'
    if target == 'log2_MT' and 'w_epi' in df.columns:
        return 'w_epi'
    return None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--prepared_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--alpha_list', default='1e-4,3e-4,1e-3')
    ap.add_argument('--l1_ratio_list', default='0.05,0.15,0.3')
    ap.add_argument('--max_iter', type=int, default=5000)
    args = ap.parse_args()

    seed_everything(args.seed)

    prepared = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_prepared_split(str(prepared / 'train.tsv'))
    val = load_prepared_split(str(prepared / 'val.tsv'))
    test = load_prepared_split(str(prepared / 'test.tsv'))

    for df in (train, val, test):
        df['sequence'] = df['sequence'].fillna('').astype(str)

    y_train = pd.to_numeric(train[args.target], errors='coerce').values.astype(float)
    y_val = pd.to_numeric(val[args.target], errors='coerce').values.astype(float)
    y_test = pd.to_numeric(test[args.target], errors='coerce').values.astype(float)

    wcol = pick_weight_col(train, args.target)
    if wcol is not None:
        w_train = pd.to_numeric(train[wcol], errors='coerce').fillna(1.0).values.astype(float)
    else:
        w_train = np.ones(len(train), dtype=float)

    alpha_list = [float(x) for x in args.alpha_list.split(',') if x.strip()]
    l1_list = [float(x) for x in args.l1_ratio_list.split(',') if x.strip()]

    best, best_cfg, best_pipe = None, None, None

    for a in alpha_list:
        for l1 in l1_list:
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(args.k, args.k), lowercase=False)),
                ('sgd', SGDRegressor(
                    loss='squared_error',
                    penalty='elasticnet',
                    alpha=float(a),
                    l1_ratio=float(l1),
                    max_iter=int(args.max_iter),
                    tol=1e-4,
                    random_state=int(args.seed),
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                )),
            ])

            pipe.fit(train['sequence'].tolist(), y_train, sgd__sample_weight=w_train)
            pv = pipe.predict(val['sequence'].tolist())
            mv = compute_metrics(y_val, pv)
            if best is None or mv['pearson'] > best['pearson']:
                best = mv
                best_cfg = {'alpha': float(a), 'l1_ratio': float(l1)}
                best_pipe = pipe

    pv = best_pipe.predict(val['sequence'].tolist())
    pt = best_pipe.predict(test['sequence'].tolist())
    mt = compute_metrics(y_test, pt)

    metrics = {
        'model': 'kmer_sgd_elasticnet',
        'target': args.target,
        'seed': int(args.seed),
        'k': int(args.k),
        'best_alpha': float(best_cfg['alpha']),
        'best_l1_ratio': float(best_cfg['l1_ratio']),
        'val_pearson': float(best['pearson']),
        'val_spearman': float(best['spearman']),
        'test_pearson': float(mt['pearson']),
        'test_spearman': float(mt['spearman']),
        'n_test': int(len(test)),
    }

    tag = out_dir.name
    write_json(metrics, out_dir / 'kmer_sgd_elasticnet.metrics.json')
    write_json(metrics, out_dir / f'{tag}.metrics.json')

    y_col_map = {
        'delta': ('y_delta', 'pred_delta', 'delta'),
        'mean': ('y_mean', 'pred_mean', 'mean'),
        'log2_WT': ('y_int', 'pred_int', 'log2_WT'),
        'log2_MT': ('y_epi', 'pred_epi', 'log2_MT'),
    }
    if args.target not in y_col_map:
        raise ValueError(f'Unsupported target for standardized outputs: {args.target}')
    y_col, p_col, src_y = y_col_map[args.target]

    out_test = test[['element_id']].copy()
    out_test[y_col] = pd.to_numeric(test[src_y], errors='coerce').values.astype(float)
    out_test[p_col] = pt
    out_test.to_csv(out_dir / f'{tag}.test_predictions.tsv', sep='\t', index=False)

    out_val = val[['element_id']].copy()
    out_val[y_col] = pd.to_numeric(val[src_y], errors='coerce').values.astype(float)
    out_val[p_col] = pv
    out_val.to_csv(out_dir / f'{tag}.val_predictions.tsv', sep='\t', index=False)

    print('DONE')
    print(metrics)


if __name__ == '__main__':
    main()
