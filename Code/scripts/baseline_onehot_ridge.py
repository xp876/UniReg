import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import compute_metrics, load_prepared_split, seed_everything, write_json, normalize_seq


DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def onehot_flatten(seqs: list[str]) -> np.ndarray:
    """Flattened one-hot encoding: (n, L*4). Unknown bases map to all-zeros."""
    if not seqs:
        return np.zeros((0, 0), dtype=np.float32)
    seqs = [normalize_seq(s) for s in seqs]
    L = len(seqs[0])
    X = np.zeros((len(seqs), L * 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"All sequences must have the same length for onehot baseline. Got {len(s)} vs {L}.")
        for j, ch in enumerate(s):
            idx = DNA_ALPHABET.get(ch)
            if idx is None:
                continue
            X[i, j * 4 + idx] = 1.0
    return X


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
    ap.add_argument('--target', required=True, help='log2_WT, log2_MT, mean, delta')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alphas', default='0.1,1,10,100')
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

    # enforce equal length
    L = int(train['sequence'].str.len().mode().iloc[0])
    train = train[train['sequence'].str.len() == L].copy()
    val = val[val['sequence'].str.len() == L].copy()
    test = test[test['sequence'].str.len() == L].copy()

    X_train = onehot_flatten(train['sequence'].tolist())
    X_val = onehot_flatten(val['sequence'].tolist())
    X_test = onehot_flatten(test['sequence'].tolist())

    y_train = pd.to_numeric(train[args.target], errors='coerce').values.astype(float)
    y_val = pd.to_numeric(val[args.target], errors='coerce').values.astype(float)
    y_test = pd.to_numeric(test[args.target], errors='coerce').values.astype(float)

    wcol = pick_weight_col(train, args.target)
    if wcol is not None:
        w_train = pd.to_numeric(train[wcol], errors='coerce').fillna(1.0).values.astype(float)
    else:
        w_train = np.ones(len(train), dtype=float)

    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    best_alpha, best_model, best_val = None, None, None
    for a in alphas:
        m = Ridge(alpha=float(a), random_state=0)
        m.fit(X_train, y_train, sample_weight=w_train)
        pv = m.predict(X_val)
        mv = compute_metrics(y_val, pv)
        if best_val is None or mv['pearson'] > best_val['pearson']:
            best_val = mv
            best_alpha = float(a)
            best_model = m

    pv = best_model.predict(X_val)
    pt = best_model.predict(X_test)
    mt = compute_metrics(y_test, pt)

    metrics = {
        'model': 'onehot_ridge',
        'target': args.target,
        'seed': int(args.seed),
        'L': int(L),
        'best_alpha': float(best_alpha),
        'val_pearson': float(best_val['pearson']),
        'val_spearman': float(best_val['spearman']),
        'test_pearson': float(mt['pearson']),
        'test_spearman': float(mt['spearman']),
        'n_test': int(len(test)),
    }

    tag = out_dir.name
    write_json(metrics, out_dir / 'onehot_ridge.metrics.json')
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
