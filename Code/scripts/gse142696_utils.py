import gzip
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd

PAIR_TO_LABEL = {
    "5p3p": "5'/3'",
    "5p5p": "5'/5'",
    "3p3p": "3'/3'",
}

# Per GEO note in mean file:
# Replicates 1 of "5'/3' MT" and "3'/3' MT" were excluded during averaging due to low barcode count and poor sample quality
BAD_REP_GLOBAL = {
    ("5'/3'", "MT"): {1},
    ("3'/3'", "MT"): {1},
}


def read_fasta_gz_as_dict(path_fa_gz: str, trim_len: int = 171) -> Dict[str, str]:
    '''
    Return dict: element_id -> sequence (first trim_len bases).

    In GSM4237954 fasta, sequences are 185bp with a constant 14bp tail;
    first 171bp is the variable insert that matches the UniReg/Plan6 convention.
    '''
    out: Dict[str, str] = {}
    name = None
    with gzip.open(path_fa_gz, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:]
                continue
            if name is None:
                continue
            seq = re.sub(r"\s+", "", line).upper().replace("U", "T")
            out[name] = seq[:trim_len]
            name = None
    return out


def _find_header_row_gz(path_gz: str, startswith: str = "name\t", max_lines: int = 200) -> int:
    with gzip.open(path_gz, "rt") as f:
        for i, line in enumerate(f):
            if line.startswith(startswith):
                return i
            if i >= max_lines:
                break
    raise ValueError(f"Could not find header row starting with '{startswith}' in {path_gz}")


def read_activity_table_mean(path_mean_gz: str) -> pd.DataFrame:
    '''
    Read the averaged ActivityRatios TSV.
    This file contains note lines before the header row.
    '''
    hdr = _find_header_row_gz(path_mean_gz, "name\t", max_lines=300)
    df = pd.read_csv(path_mean_gz, sep="\t", compression="gzip", skiprows=hdr)
    df = df.rename(columns={"name": "element_id"})
    df["element_id"] = df["element_id"].astype(str)
    return df


def read_activity_table_reps(path_reps_gz: str) -> pd.DataFrame:
    '''
    Read the individual-replicate ActivityRatios TSV.
    Header starts at the first line beginning with 'name\t'.
    '''
    hdr = _find_header_row_gz(path_reps_gz, "name\t", max_lines=20)
    df = pd.read_csv(path_reps_gz, sep="\t", compression="gzip", skiprows=hdr)
    df = df.rename(columns={"name": "element_id"})
    df["element_id"] = df["element_id"].astype(str)
    return df


def col_mean(design_label: str, wtmt: str) -> str:
    # Example: "5'/3' WT mean log2 RNA/DNA ratio"
    return f"{design_label} {wtmt} mean log2 RNA/DNA ratio"


def col_rep(design_label: str, wtmt: str, rep_i: int) -> str:
    # Example: "5'/3' WT replicate 1, log2 RNA/DNA ratio"
    return f"{design_label} {wtmt} replicate {rep_i}, log2 RNA/DNA ratio"




def get_log2_mean(df_mean: pd.DataFrame, design_label: str, wtmt: str) -> pd.Series:
    c = col_mean(design_label, wtmt)
    if c not in df_mean.columns:
        raise KeyError(f"Column not found: {c}")
    return pd.to_numeric(df_mean[c], errors="coerce")


def get_log2_reps(df_reps: pd.DataFrame, design_label: str, wtmt: str) -> Dict[int, pd.Series]:
    out: Dict[int, pd.Series] = {}
    for i in (1, 2, 3):
        c = col_rep(design_label, wtmt, i)
        if c in df_reps.columns:
            out[i] = pd.to_numeric(df_reps[c], errors="coerce")
    if not out:
        raise KeyError(f"No replicate columns found for {design_label} {wtmt}")
    return out


def apply_drop_bad_reps(reps: Dict[int, pd.Series], design_label: str, wtmt: str, drop_bad_reps: bool) -> Dict[int, pd.Series]:
    if not drop_bad_reps:
        return reps
    bad = BAD_REP_GLOBAL.get((design_label, wtmt), set())
    return {i: s for i, s in reps.items() if i not in bad}
