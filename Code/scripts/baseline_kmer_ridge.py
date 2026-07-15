import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from common import compute_metrics, load_prepared_split, seed_everything, write_json, dinuc_shuffle

def fit_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str, alphas: list[float], k_range=(3,6)):
    X_train = train_df["sequence"].astype(str).tolist()
    y_train = pd.to_numeric(train_df[target], errors="coerce").values.astype(float)
    weight_col=None
    if target=="delta" and "w_delta" in train_df.columns:
        weight_col="w_delta"
    elif target=="mean" and "w_mean" in train_df.columns:
        weight_col="w_mean"
    elif target=="log2_WT" and "w_int" in train_df.columns:
        weight_col="w_int"
    elif target=="log2_MT" and "w_epi" in train_df.columns:
        weight_col="w_epi"
    if weight_col is not None:
        w_train = pd.to_numeric(train_df[weight_col], errors="coerce").fillna(1.0).values.astype(float)
    else:
        w_train = pd.to_numeric(train_df.get("sample_weight", pd.Series(np.ones(len(train_df)))), errors="coerce").fillna(1.0).values.astype(float)

    X_val = val_df["sequence"].astype(str).tolist()
    y_val = pd.to_numeric(val_df[target], errors="coerce").values.astype(float)

    best_pipe, best_alpha, best = None, None, None
    for a in alphas:
        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=k_range, lowercase=False)),
                ("ridge", Ridge(alpha=float(a), random_state=0)),
            ]
        )
        pipe.fit(X_train, y_train, ridge__sample_weight=w_train)
        pred = pipe.predict(X_val)
        m = compute_metrics(y_val, pred)
        if (best is None) or (m["pearson"] > best["pearson"]):
            best = m
            best_alpha = float(a)
            best_pipe = pipe
    return best_pipe, best_alpha, best

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", required=True, help="Target column in prepared TSV (e.g., log2_WT, log2_MT, delta, mean).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--k", type=int, default=None, help="Shortcut for fixed k; sets kmin=kmax=k")
    ap.add_argument("--alphas", default="0.1,1,10,100")
    ap.add_argument("--shuffle_y", action="store_true", help="Negative control: shuffle training labels before fitting.")
    ap.add_argument("--dinuc_shuffle_test", action="store_true", help="Negative control: dinucleotide-shuffle TEST sequences (preserve dinuc composition, destroy motif grammar).")
    args = ap.parse_args()

    if args.k is not None:
        args.kmin = int(args.k)
        args.kmax = int(args.k)
    if args.kmin <= 0 or args.kmax <= 0 or args.kmin > args.kmax:
        raise ValueError(f"Invalid k-range: kmin={args.kmin}, kmax={args.kmax}")

    seed_everything(args.seed)

    prepared = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_prepared_split(str(prepared / "train.tsv"))
    val = load_prepared_split(str(prepared / "val.tsv"))
    test = load_prepared_split(str(prepared / "test.tsv"))

    # Optional negative control: shuffle training labels
    if args.shuffle_y:
        rng = np.random.default_rng(args.seed)
        y = pd.to_numeric(train[args.target], errors="coerce").values
        train = train.copy()
        train[args.target] = rng.permutation(y)

    # Basic cleaning
    for df in (train, val, test):
        df["sequence"] = df["sequence"].fillna("").astype(str)
    train = train[train["sequence"].str.len() >= int(args.kmin)].copy()
    val = val[val["sequence"].str.len() >= int(args.kmin)].copy()
    test = test[test["sequence"].str.len() >= int(args.kmin)].copy()

    if args.dinuc_shuffle_test:
        # Deterministic per-element shuffle (seeded by global seed + row index)
        test = test.copy()
        test["sequence"] = [dinuc_shuffle(s, seed=int(args.seed) + i) for i, s in enumerate(test["sequence"].astype(str).tolist())]

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    pipe, best_alpha, best_val = fit_and_eval(train, val, args.target, alphas, k_range=(args.kmin, args.kmax))

    y_test = pd.to_numeric(test[args.target], errors="coerce").values.astype(float)
    pred_test = pipe.predict(test["sequence"].astype(str).tolist())
    pred_val = pipe.predict(val["sequence"].astype(str).tolist())
    m_test = compute_metrics(y_test, pred_test)

    metrics = {
        "model": "kmer_ridge",
        "target": args.target,
        "seed": int(args.seed),
        "kmin": int(args.kmin),
        "kmax": int(args.kmax),
        "best_alpha": float(best_alpha),
        "val_pearson": float(best_val["pearson"]),
        "val_spearman": float(best_val["spearman"]),
        "test_pearson": float(m_test["pearson"]),
        "test_spearman": float(m_test["spearman"]),
        "n_test": int(len(test)),
    }
    tag = out_dir.name  # e.g. kmer_delta
    # metrics
    write_json(metrics, out_dir / 'kmer.metrics.json')
    write_json(metrics, out_dir / f'{tag}.metrics.json')

    # predictions (standardized y/p columns)
    y_col_map = {
        'delta': ('y_delta', 'pred_delta', 'delta'),
        'mean': ('y_mean', 'pred_mean', 'mean'),
        'log2_WT': ('y_int', 'pred_int', 'log2_WT'),
        'log2_MT': ('y_epi', 'pred_epi', 'log2_MT'),
    }
    if args.target not in y_col_map:
        raise ValueError(f'Unsupported target for standardized outputs: {args.target}')
    y_col, p_col, src_y = y_col_map[args.target]

    pred_df = test[['element_id']].copy()
    if 'sequence' in test.columns:
        pred_df['sequence'] = test['sequence'].values
    for col in ['log2_WT','log2_MT','delta','mean']:
        if col in test.columns and col not in pred_df.columns:
            pred_df[col] = test[col].values

    pred_df[y_col] = pd.to_numeric(test[src_y], errors='coerce').values.astype(float)
    pred_df[p_col] = pred_test

    # legacy + model-tagged
    legacy = pred_df.copy()
    legacy['pred'] = pred_test
    legacy.to_csv(out_dir / 'kmer.test_predictions.tsv', sep='	', index=False)
    pred_df.to_csv(out_dir / f'{tag}.test_predictions.tsv', sep='	', index=False)

    # validation predictions (useful for stacking/fusion)
    yv = pd.to_numeric(val[src_y], errors='coerce').values.astype(float)
    val_df_out = val[['element_id']].copy()
    if 'sequence' in val.columns:
        val_df_out['sequence'] = val['sequence'].values
    for col in ['log2_WT','log2_MT','delta','mean']:
        if col in val.columns and col not in val_df_out.columns:
            val_df_out[col] = val[col].values
    val_df_out[y_col] = yv
    val_df_out[p_col] = pred_val
    val_df_out.to_csv(out_dir / f'{tag}.val_predictions.tsv', sep='	', index=False)


    print('DONE')
    print(metrics)


if __name__ == "__main__":
    main()
