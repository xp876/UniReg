import argparse
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gse142696_utils import (
    PAIR_TO_LABEL,
    read_fasta_gz_as_dict,
    read_activity_table_mean,
    read_activity_table_reps,
    get_log2_mean,
    get_log2_reps,
    apply_drop_bad_reps,
)


def _pow2_safe(x: float) -> float:
    # convert log2 ratio -> ratio, preserving NaNs
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(2.0 ** float(x))


def _write_unireg_zip(rows: pd.DataFrame, out_zip: str, root: str) -> None:
    '''
    Create a UniReg-style zip expected by Plan6 scripts.
    It must contain {root}/train.csv, val.csv, test.csv.
    We store all rows in train.csv, and keep val/test as empty (header only).
    '''
    rows = rows.copy()
    # canonical columns
    if "seq" in rows.columns and "sequence" not in rows.columns:
        rows = rows.rename(columns={"seq": "sequence"})
    # ensure some ordering
    cols = [c for c in ["element_id", "sequence", "context", "replicate", "activity_raw"] if c in rows.columns]
    cols += [c for c in rows.columns if c not in cols]
    rows = rows[cols]

    root = root.rstrip("/") + "/"
    out_zip = str(out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # train.csv
        train_bytes = rows.to_csv(index=False).encode("utf-8")
        z.writestr(root + "train.csv", train_bytes)

        # empty val/test with headers
        empty = rows.head(0)
        z.writestr(root + "val.csv", empty.to_csv(index=False).encode("utf-8"))
        z.writestr(root + "test.csv", empty.to_csv(index=False).encode("utf-8"))


def build_exports(
    pair: str,
    elements_fa: str,
    activity_reps_tsv: str,
    activity_mean_tsv: str,
    out_dir: str,
    drop_bad_reps: bool = True,
    trim_len: int = 171,
) -> Tuple[str, str]:
    '''
    Build (formatB_agg_only.zip, formatA_all_replicates.zip) for a single lentiMPRA pair.
    pair: 5p3p / 5p5p / 3p3p
    '''
    if pair not in PAIR_TO_LABEL:
        raise ValueError(f"Unknown pair={pair}. Choose from {list(PAIR_TO_LABEL.keys())}")
    design_label = PAIR_TO_LABEL[pair]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) sequences
    seq_map = read_fasta_gz_as_dict(elements_fa, trim_len=trim_len)

    # 2) activity tables
    df_reps = read_activity_table_reps(activity_reps_tsv)
    df_mean = read_activity_table_mean(activity_mean_tsv)

    # align on element_id
    df = pd.DataFrame({"element_id": df_reps["element_id"].astype(str)})
    df = df.merge(df_mean[["element_id"]], on="element_id", how="inner")
    df = df[df["element_id"].isin(seq_map.keys())].copy()
    df["sequence"] = df["element_id"].map(seq_map)

    # mean log2
    wt_mean = get_log2_mean(df_mean.set_index("element_id").loc[df["element_id"]].reset_index(), design_label, "WT")
    mt_mean = get_log2_mean(df_mean.set_index("element_id").loc[df["element_id"]].reset_index(), design_label, "MT")

    # replicates log2
    wt_reps = get_log2_reps(df_reps.set_index("element_id").loc[df["element_id"]].reset_index(), design_label, "WT")
    mt_reps = get_log2_reps(df_reps.set_index("element_id").loc[df["element_id"]].reset_index(), design_label, "MT")

    wt_reps = apply_drop_bad_reps(wt_reps, design_label, "WT", drop_bad_reps)
    mt_reps = apply_drop_bad_reps(mt_reps, design_label, "MT", drop_bad_reps)

    # 3) FormatB (agg only): WT_agg + MT_agg
    fmtB_rows = []
    for idx, eid in enumerate(df["element_id"].tolist()):
        seq = df.at[df.index[idx], "sequence"]
        lw = float(wt_mean.iloc[idx]) if idx < len(wt_mean) else float("nan")
        lm = float(mt_mean.iloc[idx]) if idx < len(mt_mean) else float("nan")
        if not np.isfinite(lw) or not np.isfinite(lm):
            continue
        fmtB_rows.append({"element_id": eid, "sequence": seq, "context": "INT", "replicate": "WT_agg", "activity_raw": _pow2_safe(lw)})
        fmtB_rows.append({"element_id": eid, "sequence": seq, "context": "EPI", "replicate": "MT_agg", "activity_raw": _pow2_safe(lm)})

    fmtB = pd.DataFrame(fmtB_rows)
    # ensure paired
    counts = fmtB.groupby(["element_id", "replicate"]).size().unstack(fill_value=0)
    keep = counts[(counts.get("WT_agg", 0) > 0) & (counts.get("MT_agg", 0) > 0)].index.astype(str)
    fmtB = fmtB[fmtB["element_id"].isin(keep)].copy()

    # 4) FormatA (replicates + agg)
    fmtA_rows = []
    # create maps from eid->row index for mean table slices
    eid_to_pos = {eid: i for i, eid in enumerate(df["element_id"].tolist())}

    for eid in keep.tolist():
        pos = eid_to_pos[eid]
        seq = df.loc[df["element_id"] == eid, "sequence"].iloc[0]

        lw = float(wt_mean.iloc[pos])
        lm = float(mt_mean.iloc[pos])

        # agg
        fmtA_rows.append({"element_id": eid, "sequence": seq, "context": "INT", "replicate": "WT_agg", "activity_raw": _pow2_safe(lw)})
        fmtA_rows.append({"element_id": eid, "sequence": seq, "context": "EPI", "replicate": "MT_agg", "activity_raw": _pow2_safe(lm)})

        # reps
        for i, s in wt_reps.items():
            v = float(s.iloc[pos])
            if np.isfinite(v):
                fmtA_rows.append({"element_id": eid, "sequence": seq, "context": "INT", "replicate": f"WT{i}", "activity_raw": _pow2_safe(v)})
        for i, s in mt_reps.items():
            v = float(s.iloc[pos])
            if np.isfinite(v):
                fmtA_rows.append({"element_id": eid, "sequence": seq, "context": "EPI", "replicate": f"MT{i}", "activity_raw": _pow2_safe(v)})

    fmtA = pd.DataFrame(fmtA_rows)

    # write zips
    zB = str(out_dir / "formatB_agg_only.zip")
    zA = str(out_dir / "formatA_all_replicates.zip")
    _write_unireg_zip(fmtB, zB, root="data_formatB")
    _write_unireg_zip(fmtA, zA, root="data_formatA")

    # report
    n_elem = fmtB["element_id"].nunique()
    print(f"[GSE142696] pair={pair} design='{design_label}' elements(paired)={n_elem}")
    print("Wrote:", zB)
    print("Wrote:", zA)
    return zB, zA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="5p3p | 5p5p | 3p3p")
    ap.add_argument("--elements_fa", required=True)
    ap.add_argument("--activity_reps_tsv", required=True, help="GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz")
    ap.add_argument("--activity_mean_tsv", required=True, help="GSE142696_9MPRA.ActivityRatios.tsv.gz (averaged)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop_bad_reps", action="store_true", help="Drop globally bad replicates noted in GEO mean file")
    ap.add_argument("--keep_bad_reps", action="store_true", help="Override: keep all replicates")
    ap.add_argument("--trim_len", type=int, default=171)
    args = ap.parse_args()

    drop = True
    if args.keep_bad_reps:
        drop = False
    if args.drop_bad_reps:
        drop = True

    build_exports(
        pair=args.pair,
        elements_fa=args.elements_fa,
        activity_reps_tsv=args.activity_reps_tsv,
        activity_mean_tsv=args.activity_mean_tsv,
        out_dir=args.out_dir,
        drop_bad_reps=drop,
        trim_len=int(args.trim_len),
    )


if __name__ == "__main__":
    main()
