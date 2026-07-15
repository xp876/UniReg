import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_control_spans(element_id: str) -> List[Tuple[int, int]]:
    """Parse motif insertion spans from control element_id.

    Example:
      C:SLEA...|21:V_HNF3ALPHA_Q6:TGTTTGCTTTG;35:V_AHRARNT_02:GGGG...

    Returns list of (start, end_exclusive) spans.
    """
    s = str(element_id)
    if "|" not in s:
        return []
    ann = s.split("|", 1)[1]
    spans = []
    for item in ann.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            continue
        try:
            pos = int(parts[0])
        except Exception:
            continue
        motif_seq = parts[-1].strip().upper()
        L = len(motif_seq)
        if L <= 0:
            continue
        spans.append((pos, pos + L))
    return spans


def in_any_span(i: int, spans: List[Tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= i < b:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism_summary", required=True, help="ism_summary_all.tsv from ism_delta_gb.py")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.ism_summary, sep="\t")
    df["element_id"] = df["element_id"].astype(str)

    # keep controls only
    df = df[df["element_id"].str.startswith("C:")].copy()
    if df.empty:
        raise SystemExit("No controls found in ISM summary. Run ISM with --only_controls or ensure controls in picked set.")

    rows = []
    for eid, sub in df.groupby("element_id"):
        spans = parse_control_spans(eid)
        if not spans:
            continue
        # scores per position
        sub = sub.sort_values("pos")
        pos = sub["pos"].values.astype(int)
        scores = sub["importance"].values.astype(float)
        L = int(pos.max() + 1)
        y = np.zeros(L, dtype=int)
        s = np.zeros(L, dtype=float)
        s[pos] = scores
        for i in range(L):
            y[i] = 1 if in_any_span(i, spans) else 0
        if y.sum() == 0 or y.sum() == len(y):
            continue

        # AUPRC / AUROC
        apv = float(average_precision_score(y, s))
        try:
            auc = float(roc_auc_score(y, s))
        except Exception:
            auc = float("nan")

        # top-k overlap
        k = int(args.topk)
        top_idx = np.argsort(-s)[:k]
        overlap = float(np.mean([in_any_span(int(i), spans) for i in top_idx]))

        rows.append({
            "element_id": eid,
            "n_spans": int(len(spans)),
            "span_total_bp": int(sum(b-a for a,b in spans)),
            "auprc": apv,
            "auroc": auc,
            f"top{ k }_overlap": overlap,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    # aggregate stats
    summary = {
        "n_controls": int(len(out)),
        "auprc_mean": float(out["auprc"].mean()),
        "auprc_std": float(out["auprc"].std(ddof=0)),
        "auroc_mean": float(out["auroc"].mean()),
        "auroc_std": float(out["auroc"].std(ddof=0)),
    }
    print("Wrote:", out_path)
    print(summary)


if __name__ == "__main__":
    main()
