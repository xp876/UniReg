import argparse
import re
import numpy as np
import pandas as pd

from common import load_zip_splits, normalize_seq, log2_safe


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seq" in df.columns and "sequence" not in df.columns:
        df = df.rename(columns={"seq": "sequence"})
    df["sequence"] = df["sequence"].astype(str).map(normalize_seq)
    df["log2_raw"] = pd.to_numeric(df["activity_raw"], errors="coerce").map(log2_safe)
    df["context"] = df["context"].astype(str).str.upper()
    df["replicate"] = df["replicate"].astype(str)
    df["element_id"] = df["element_id"].astype(str)
    return df


def _var_row(v: np.ndarray) -> float:
    v = v.astype(float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.var(v, ddof=1))


def _rep_indices(cols: list, prefix: str) -> list:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    out = []
    for c in cols:
        m = pat.match(str(c))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_zip", required=True, help="FormatA_all_replicates.zip")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--eps", type=float, default=1e-3, help="Stabilizer for weights: 1/(var+eps)")
    ap.add_argument("--clip", type=float, default=50.0, help="Clip weights to [1/clip, clip]")
    ap.add_argument("--int_prefix", default="WT")
    ap.add_argument("--epi_prefix", default="MT")
    args = ap.parse_args()

    df = pd.concat(load_zip_splits(args.data_zip).values(), ignore_index=True)
    df = _prep(df)

    wt = df[df["context"] == "INT"].copy()
    mt = df[df["context"] == "EPI"].copy()

    # find available replicate indices
    wt_idx = _rep_indices(wt["replicate"].unique().tolist(), args.int_prefix)
    mt_idx = _rep_indices(mt["replicate"].unique().tolist(), args.epi_prefix)
    common = sorted(set(wt_idx).intersection(mt_idx))
    if len(common) < 2:
        print("WARN: fewer than 2 matched replicate indices for delta variance. common=", common)

    wt = wt[wt["replicate"].isin([f"{args.int_prefix}{i}" for i in wt_idx])].copy()
    mt = mt[mt["replicate"].isin([f"{args.epi_prefix}{i}" for i in mt_idx])].copy()

    wt_pv = wt.pivot_table(index="element_id", columns="replicate", values="log2_raw", aggfunc="first")
    mt_pv = mt.pivot_table(index="element_id", columns="replicate", values="log2_raw", aggfunc="first")

    idx = wt_pv.index.intersection(mt_pv.index)
    wt_pv = wt_pv.loc[idx]
    mt_pv = mt_pv.loc[idx]

    var_wt = wt_pv.apply(lambda r: _var_row(r.values), axis=1)
    var_mt = mt_pv.apply(lambda r: _var_row(r.values), axis=1)

    # delta replicates for matched indices
    delta_cols = []
    for i in common:
        a = wt_pv.get(f"{args.int_prefix}{i}")
        b = mt_pv.get(f"{args.epi_prefix}{i}")
        if a is None or b is None:
            continue
        delta_cols.append(a - b)
    if not delta_cols:
        raise SystemExit("No matched WT/MT replicates found; cannot compute delta variance")

    delta_mat = pd.concat(delta_cols, axis=1)
    delta_mat.columns = [f"delta{i}" for i in common[: delta_mat.shape[1]]]
    var_delta = delta_mat.apply(lambda r: _var_row(r.values), axis=1)

    # mean replicates
    mean_cols = []
    for i in common:
        a = wt_pv.get(f"{args.int_prefix}{i}")
        b = mt_pv.get(f"{args.epi_prefix}{i}")
        if a is None or b is None:
            continue
        mean_cols.append(0.5 * (a + b))
    mean_mat = pd.concat(mean_cols, axis=1)
    mean_mat.columns = [f"mean{i}" for i in common[: mean_mat.shape[1]]]
    var_mean = mean_mat.apply(lambda r: _var_row(r.values), axis=1)

    eps = float(args.eps)
    clip = float(args.clip)

    def w_from_var(v: pd.Series) -> pd.Series:
        vv = v.copy()
        med = float(np.nanmedian(vv.values.astype(float)))
        vv = vv.fillna(med if np.isfinite(med) else 0.0)
        w = 1.0 / (vv + eps)
        w = w.clip(lower=1.0 / clip, upper=clip)
        return w

    out = pd.DataFrame({
        "element_id": idx.astype(str),
        "var_wt": var_wt.values,
        "var_mt": var_mt.values,
        "var_delta": var_delta.values,
        "var_mean": var_mean.values,
    })
    out["w_int"] = w_from_var(out["var_wt"])
    out["w_epi"] = w_from_var(out["var_mt"])
    out["w_delta"] = w_from_var(out["var_delta"])
    out["w_mean"] = w_from_var(out["var_mean"])

    out.to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)
    print({"n": int(out.shape[0]), "wt_rep_indices": wt_idx, "mt_rep_indices": mt_idx, "matched": common})


if __name__ == "__main__":
    main()
