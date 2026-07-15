import os
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def ensure_out_dir(out_dir: str):
    """Create output directory if needed and return it as Path."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int = 42):
    """Alias for backward-compatibility."""
    return seed_everything(seed)


def normalize_seq(seq: str) -> str:
    """Uppercase, U->T, non-ACGT -> N."""
    if not isinstance(seq, str):
        return ""
    s = seq.strip().upper().replace("U", "T")
    s = "".join([c if c in "ACGT" else "N" for c in s])
    return s


def trim_pad_center(seq: str, L: int, pad: str = "N") -> str:
    """Center crop or pad to length L."""
    s = normalize_seq(seq)
    if L <= 0:
        return s
    if len(s) == L:
        return s
    if len(s) > L:
        start = (len(s) - L) // 2
        return s[start : start + L]
    # pad
    total = L - len(s)
    left = total // 2
    right = total - left
    return (pad * left) + s + (pad * right)


def pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    p = pearsonr(y_true[mask], y_pred[mask])[0]
    s = spearmanr(y_true[mask], y_pred[mask])[0]
    return float(p), float(s)


def infer_dataset_root(names: List[str]) -> str:
    """Given zip members, find the common root directory containing train/val/test.csv."""
    for prefix in ["data_formatB/", "data_formatA/"]:
        if any(n.startswith(prefix) for n in names):
            return prefix
    # generic: find first path that contains train.csv
    for n in names:
        if n.endswith("train.csv"):
            # root is everything up to train.csv
            return n[: -len("train.csv")]
    raise ValueError("Could not find train.csv in zip")


def read_split_csv_from_zip(zpath: str, split: str) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        root = infer_dataset_root(names)
        member = f"{root}{split}.csv"
        if member not in names:
            raise FileNotFoundError(f"{member} not found in {zpath}")
        with z.open(member) as f:
            df = pd.read_csv(f)
    return df


def load_zip_splits(zpath: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for sp in ["train", "val", "test"]:
        out[sp] = read_split_csv_from_zip(zpath, sp)
    return out


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify sequence column name to 'sequence', and ensure required columns exist if possible."""
    df = df.copy()
    if "sequence" not in df.columns and "seq" in df.columns:
        df = df.rename(columns={"seq": "sequence"})
    return df


REQUIRED_BASE_COLS = ["element_id", "sequence", "context"]


def ensure_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    df = canonicalize_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns={df.columns.tolist()}")
    return df


def filter_context(df: pd.DataFrame, context: str = "INT") -> pd.DataFrame:
    if "context" not in df.columns:
        return df
    return df[df["context"].astype(str).str.upper() == context.upper()].copy()


def choose_trim_length(train_df: pd.DataFrame, default: Optional[int] = None) -> int:
    lens = train_df["sequence"].astype(str).map(len)
    if lens.empty:
        return default or 0
    mode_len = int(lens.value_counts().idxmax())
    if default is not None:
        return int(default)
    return mode_len


def apply_trim(df: pd.DataFrame, L: int) -> pd.DataFrame:
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).map(lambda s: trim_pad_center(s, L))
    df["len"] = df["sequence"].astype(str).map(len)
    return df


def add_sample_weight_per_element(df: pd.DataFrame, element_col: str = "element_id") -> pd.DataFrame:
    """Make sum of weights per element ~= 1 (helps FormatA replicate-heavy)."""
    df = df.copy()
    counts = df[element_col].value_counts()
    df["sample_weight"] = df[element_col].map(lambda x: 1.0 / float(counts.get(x, 1)))
    return df


@dataclass
class PreparedInfo:
    trim_to: int
    n_rows: Dict[str, int]
    n_element_id: Dict[str, int]
    context_counts: Dict[str, int]
    target: str
    target_stats: Dict[str, Dict[str, float]]
    missing_element_ids_vs_splits: Dict[str, int]


def compute_target_stats(df: pd.DataFrame, target: str) -> Dict[str, float]:
    y = pd.to_numeric(df[target], errors="coerce")
    return {
        "n": int(y.notna().sum()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
        "min": float(y.min()),
        "max": float(y.max()),
    }


def read_json(path: str):
    """Read JSON file helper used by posthoc/transfer scripts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_splits_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # expected: {train:[...], val:[...], test:[...]}
    out = {}
    for sp in ["train", "val", "test"]:
        if sp not in data:
            raise ValueError(f"splits json missing key '{sp}'")
        out[sp] = list(map(str, data[sp]))
    return out


# --- Compatibility helpers for Step4 scripts ---

def load_prepared_split(path_or_dir: str, split: str | None = None) -> pd.DataFrame:
    """Load a prepared split TSV (train/val/test).

    Accepts either:
      - a direct TSV file path (e.g. ".../train.tsv"); or
      - a prepared_dir + split name (e.g. (".../prepared", "train")).
    """

    path = path_or_dir
    if split is not None:
        path = os.path.join(path_or_dir, f"{split}.tsv")
    df = pd.read_csv(path, sep='	')
    # Canonicalize sequence column name
    if 'seq' in df.columns and 'sequence' not in df.columns:
        df = df.rename(columns={'seq':'sequence'})
    if 'sequence' not in df.columns:
        raise ValueError(f'Missing sequence column in {path}; columns={df.columns.tolist()}')
    if 'context' not in df.columns:
        df['context'] = 'INT'
    if 'replicate' not in df.columns:
        df['replicate'] = 'agg'
    if 'sample_weight' not in df.columns:
        df['sample_weight'] = 1.0
    return df

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, s = pearson_spearman(y_true, y_pred)
    return {'pearson': float(p), 'spearman': float(s)}


# --- Plan3 helpers ---

def log2_safe(x: float, eps: float = 1e-8) -> float:
    """Safe log2 for positive ratios; clamps at eps."""
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    v = max(v, eps)
    return float(np.log2(v))

def element_level_aggregate(df: pd.DataFrame, cols: list[str], element_col: str = "element_id") -> pd.DataFrame:
    """Aggregate row-level replicates to element-level by mean for specified cols."""
    g = df.groupby(element_col, as_index=False)[cols].mean()
    return g


def maybe_clip_quantile(y: np.ndarray, clip_q: float) -> np.ndarray:
    """Clip values to the [q, 1-q] quantile range.

    This is a small but useful stabilizer for heavy-tailed ratio targets such as
    delta = log2(WT/MT). If *clip_q* <= 0, returns the input unchanged.

    NaNs are preserved.
    """
    y = np.asarray(y).astype(float)
    q = float(clip_q)
    if (not np.isfinite(q)) or q <= 0.0:
        return y
    q = min(q, 0.49)  # avoid degenerate quantiles
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return y
    lo = float(np.quantile(y[mask], q))
    hi = float(np.quantile(y[mask], 1.0 - q))
    out = y.copy()
    out[mask] = np.clip(out[mask], lo, hi)
    return out


def dinuc_shuffle(seq: str, seed: int | None = None) -> str:
    """Return a dinucleotide-preserving shuffle of *seq* (directed Eulerian trail).

    This preserves counts of all adjacent pairs (including N or other symbols),
    and therefore preserves mono- and dinucleotide composition.

    Deterministic if *seed* is provided.

    Reference idea: represent the sequence as a directed multigraph of adjacent pairs
    and sample a random Eulerian trail by shuffling outgoing edges then running
    Hierholzer's algorithm.
    """
    if seq is None:
        return ''
    s = str(seq)
    if len(s) <= 2:
        return s

    import numpy as _np
    rng = _np.random.default_rng(seed)

    # Build adjacency lists of outgoing edges.
    adj: dict[str, list[str]] = {}
    for a, b in zip(s[:-1], s[1:]):
        adj.setdefault(a, []).append(b)
        # ensure node exists
        adj.setdefault(b, adj.get(b, []))

    # Shuffle outgoing edges to randomize the Eulerian trail.
    for k in list(adj.keys()):
        if adj[k]:
            rng.shuffle(adj[k])

    start = s[0]
    stack = [start]
    circuit: list[str] = []

    while stack:
        v = stack[-1]
        if adj.get(v):
            nxt = adj[v].pop()
            stack.append(nxt)
        else:
            circuit.append(stack.pop())

    circuit.reverse()
    out = ''.join(circuit)
    # Safety: length must match
    if len(out) != len(s):
        return s
    return out
