import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_zip_splits, normalize_seq, log2_safe, write_json


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seq" in df.columns and "sequence" not in df.columns:
        df = df.rename(columns={"seq": "sequence"})
    df["sequence"] = df["sequence"].astype(str).map(normalize_seq)
    df["log2_raw"] = pd.to_numeric(df["activity_raw"], errors="coerce").map(log2_safe)
    df["context"] = df["context"].astype(str).str.upper()
    df["replicate"] = df["replicate"].astype(str)
    df["eid_prefix"] = df["element_id"].astype(str).map(element_prefix)
    return df


def _rep_indices(reps: list[str], prefix: str) -> list[int]:
    out = []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    for r in reps:
        m = pat.match(str(r))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _pairwise_corr(mat: pd.DataFrame) -> dict:
    reps = list(mat.columns)
    out = {}
    for a, b in itertools.combinations(reps, 2):
        x = mat[a].values.astype(float)
        y = mat[b].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 8:
            continue
        out[f"{a}__{b}"] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return out


def _agg_vs_meanrep(agg_series: pd.Series, rep_mat: pd.DataFrame) -> float:
    rep_mean = rep_mat.mean(axis=1)
    joined = pd.concat([agg_series, rep_mean], axis=1, keys=["agg", "mean_rep"]).dropna()
    if len(joined) < 8:
        return float("nan")
    return float(np.corrcoef(joined["agg"].values, joined["mean_rep"].values)[0, 1])


def _summ(pw: dict) -> float:
    if not pw:
        return float("nan")
    vals = [v for v in pw.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _context_report(df: pd.DataFrame, ctx: str, rep_prefix: str):
    sub = df[df["context"] == ctx].copy()
    rep_ids = _rep_indices(sub["replicate"].unique().tolist(), rep_prefix)
    rep_names = [f"{rep_prefix}{i}" for i in rep_ids]

    sub_r = sub[sub["replicate"].isin(rep_names)].copy()
    pv = sub_r.pivot_table(index="element_id", columns="replicate", values="log2_raw", aggfunc="first")

    pw = _pairwise_corr(pv.dropna())
    mean_pw = _summ(pw)

    agg = sub[sub["replicate"].isin([f"{rep_prefix}_agg", f"{rep_prefix.upper()}_agg"])].copy()
    agg_pv = agg.set_index("element_id")["log2_raw"] if len(agg) else pd.Series(dtype=float)
    corr_agg_mean = _agg_vs_meanrep(agg_pv, pv) if len(agg_pv) else float("nan")

    by_pref = {}
    for pref, sub2 in sub_r.groupby("eid_prefix"):
        pv2 = sub2.pivot_table(index="element_id", columns="replicate", values="log2_raw", aggfunc="first")
        pw2 = _pairwise_corr(pv2.dropna())
        by_pref[pref] = {
            "n_elements_reps": int(pv2.dropna().shape[0]),
            "pairwise_mean_pearson": _summ(pw2),
        }

    return {
        "rep_indices": rep_ids,
        "n_elements_reps": int(pv.dropna().shape[0]),
        "pairwise_mean_pearson": mean_pw,
        "corr_agg_vs_meanrep": corr_agg_mean,
        "by_element_prefix": by_pref,
    }, pv, agg_pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_zip", required=True, help="FormatA_all_replicates.zip")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--int_prefix", default="WT")
    ap.add_argument("--epi_prefix", default="MT")
    args = ap.parse_args()

    df = pd.concat(load_zip_splits(args.data_zip).values(), ignore_index=True)
    df = _prep(df)

    rep_int, wt_pv, wt_agg = _context_report(df, "INT", args.int_prefix)
    rep_epi, mt_pv, mt_agg = _context_report(df, "EPI", args.epi_prefix)

    # align rep mats
    idx = wt_pv.index.intersection(mt_pv.index)
    wt_pv = wt_pv.loc[idx]
    mt_pv = mt_pv.loc[idx]

    common_idx = sorted(set(_rep_indices(list(wt_pv.columns), args.int_prefix)).intersection(_rep_indices(list(mt_pv.columns), args.epi_prefix)))
    delta_cols = {}
    for i in common_idx:
        a = wt_pv.get(f"{args.int_prefix}{i}")
        b = mt_pv.get(f"{args.epi_prefix}{i}")
        if a is None or b is None:
            continue
        delta_cols[f"delta{i}"] = a - b
    delta_mat = pd.DataFrame(delta_cols, index=wt_pv.index)

    delta_pw = _pairwise_corr(delta_mat.dropna())
    delta_mean_pw = _summ(delta_pw)

    # delta agg
    corr_delta_agg_meanrep = float("nan")
    if len(wt_agg) and len(mt_agg):
        common_eids = wt_agg.index.intersection(mt_agg.index)
        delta_agg = (wt_agg.loc[common_eids] - mt_agg.loc[common_eids])
        corr_delta_agg_meanrep = _agg_vs_meanrep(delta_agg, delta_mat)

    # delta by element prefix (R/A/C etc)
    e2pref = df.drop_duplicates("element_id").set_index("element_id")["eid_prefix"].to_dict()
    by_pref = {}
    for pref in sorted(set(e2pref.values())):
        eids = [eid for eid, p in e2pref.items() if p == pref]
        subm = delta_mat.loc[delta_mat.index.intersection(eids)].dropna()
        pw = _pairwise_corr(subm)
        by_pref[pref] = {
            "n_elements_reps": int(subm.shape[0]),
            "pairwise_mean_pearson": _summ(pw),
        }

    out = {
        "dataset": str(args.data_zip),
        "INT": rep_int,
        "EPI": rep_epi,
        "DELTA": {
            "matched_rep_indices": common_idx,
            "n_elements_reps": int(delta_mat.dropna().shape[0]),
            "pairwise_mean_pearson": float(delta_mean_pw),
            "corr_agg_vs_meanrep": float(corr_delta_agg_meanrep),
            "by_element_prefix": by_pref,
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, args.out_json)
    print("Wrote:", args.out_json)


if __name__ == "__main__":
    main()
