import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import compute_metrics, write_json


def load_pred(path: Path, y_col: str, p_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    if 'element_id' not in df.columns:
        raise SystemExit(f'missing element_id in {path}')
    if y_col not in df.columns or p_col not in df.columns:
        raise SystemExit(f'missing {y_col}/{p_col} in {path}')
    df = df[['element_id', y_col, p_col]].copy()
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
    df = df.dropna(subset=[y_col, p_col]).copy()
    return df


def fit_weight_mse(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    d = p1 - p2
    denom = float(np.sum(d * d))
    if denom <= 1e-12:
        return 0.5
    w = float(np.sum((y - p2) * d) / denom)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_pred1', required=True)
    ap.add_argument('--val_pred2', required=True)
    ap.add_argument('--test_pred1', required=True)
    ap.add_argument('--test_pred2', required=True)
    ap.add_argument('--y_col', default='y_delta')
    ap.add_argument('--p_col1', default='pred_delta')
    ap.add_argument('--p_col2', default='pred_delta')
    ap.add_argument('--constrain_01', action='store_true', help='Clamp weight to [0,1]')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--name', default='fused')
    ap.add_argument('--out_pred_col', default='pred_delta', help='Name of fused prediction column in output TSV (default: pred_delta)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = load_pred(Path(args.val_pred1), args.y_col, args.p_col1)
    v2 = load_pred(Path(args.val_pred2), args.y_col, args.p_col2)
    vm = pd.merge(v1, v2, on='element_id', suffixes=('_1', '_2'))

    y = vm[f'{args.y_col}_1'].values.astype(float)
    p1 = vm[f'{args.p_col1}_1'].values.astype(float)
    p2 = vm[f'{args.p_col2}_2'].values.astype(float)

    w = fit_weight_mse(y, p1, p2)
    if args.constrain_01:
        w = float(max(0.0, min(1.0, w)))

    # Evaluate on val
    pv = w * p1 + (1.0 - w) * p2
    m_val = compute_metrics(y, pv)

    # Apply on test
    t1 = load_pred(Path(args.test_pred1), args.y_col, args.p_col1)
    t2 = load_pred(Path(args.test_pred2), args.y_col, args.p_col2)
    tm = pd.merge(t1, t2, on='element_id', suffixes=('_1', '_2'))

    yt = tm[f'{args.y_col}_1'].values.astype(float)
    tp1 = tm[f'{args.p_col1}_1'].values.astype(float)
    tp2 = tm[f'{args.p_col2}_2'].values.astype(float)
    pt = w * tp1 + (1.0 - w) * tp2

    m_test = compute_metrics(yt, pt)

    # Standardize output so downstream summarizers can treat this like a normal model.
    out_pred = pd.DataFrame({'element_id': tm['element_id'].values, args.y_col: yt})
    out_pred[args.out_pred_col] = pt
    # keep provenance/debug columns
    out_pred['pred_1'] = tp1
    out_pred['pred_2'] = tp2
    out_pred['w'] = w
    out_pred.to_csv(out_dir / f'{args.name}.test_predictions.tsv', sep='\t', index=False)

    write_json({
        'val': m_val,
        'test': m_test,
        'weight_w_for_model1': w,
        'val_pred1': str(args.val_pred1),
        'val_pred2': str(args.val_pred2),
        'test_pred1': str(args.test_pred1),
        'test_pred2': str(args.test_pred2),
    }, out_dir / f'{args.name}.metrics.json')

    print('Wrote:', out_dir)
    print('w:', w)
    print('Val:', m_val)
    print('Test:', m_test)


if __name__ == '__main__':
    main()
