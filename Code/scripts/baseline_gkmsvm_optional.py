"""Optional baseline using LS-GKM (gkm-SVM).

This is intentionally OPTIONAL because LS-GKM is an external C++ dependency.
If `gkmtrain` and `gkmpredict` are in PATH, this script runs and produces:
- gkmsvm_optional.metrics.json
- gkmsvm_optional.test_predictions.tsv  (columns: element_id, y_delta, pred_delta)

We train an LS-GKM *classifier* by binarizing delta (top-q vs bottom-q), then
use the decision scores as a *ranking/regression proxy* and evaluate correlation
with the continuous delta.

That setup is common in MPRA papers as a strong gapped-kmer baseline when a
full gkm-kernel ridge implementation is not available.

Usage:
  python baseline_gkmsvm_optional.py --prepared_dir ... --out_dir ...

Install LS-GKM:
  https://github.com/DongwonLee/lsgkm
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_prepared_split, write_json, compute_metrics


def write_fasta(df: pd.DataFrame, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            eid = str(row["element_id"]).replace(" ", "_")
            seq = str(row["sequence"]).upper()
            f.write(f">{eid}\n{seq}\n")


def _read_scores(path: Path) -> np.ndarray:
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            scores.append(float(parts[-1]))
    return np.asarray(scores, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--q", type=float, default=0.2, help="Quantile for binarization (top q vs bottom q)")
    ap.add_argument("--gkmtrain", default="gkmtrain")
    ap.add_argument("--gkmpredict", default="gkmpredict")
    args = ap.parse_args()

    if shutil.which(args.gkmtrain) is None or shutil.which(args.gkmpredict) is None:
        print("LS-GKM binaries not found in PATH. Skipping gkm-SVM baseline.")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "model": "gkmsvm_optional",
                "status": "skipped",
                "reason": "missing_binaries",
                "gkmtrain": args.gkmtrain,
                "gkmpredict": args.gkmpredict,
            },
            out_dir / "gkmsvm_optional.metrics.json",
        )
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_prepared_split(str(Path(args.prepared_dir) / "train.tsv"))
    test = load_prepared_split(str(Path(args.prepared_dir) / "test.tsv"))

    if "delta" not in train.columns:
        raise SystemExit("prepared_dir/train.tsv missing 'delta' column")

    y_train = pd.to_numeric(train["delta"], errors="coerce")
    lo = float(y_train.quantile(args.q))
    hi = float(y_train.quantile(1 - args.q))

    train_pos = train[y_train >= hi].copy()
    train_neg = train[y_train <= lo].copy()

    fasta_pos = out_dir / "train_pos.fa"
    fasta_neg = out_dir / "train_neg.fa"
    write_fasta(train_pos, fasta_pos)
    write_fasta(train_neg, fasta_neg)

    model_prefix = out_dir / "gkmsvm_model"

    # NOTE: gkmtrain expects an *output prefix* and typically writes
    #   <prefix>.model.txt
    # but gkmpredict expects the *model file path*.
    cmd_train = [args.gkmtrain, str(fasta_pos), str(fasta_neg), str(model_prefix)]
    print("Running:", " ".join(cmd_train))
    try:
        subprocess.check_call(cmd_train)
    except subprocess.CalledProcessError as e:
        # Optional baseline: do not fail the whole pipeline.
        write_json(
            {
                "model": "gkmsvm_optional",
                "status": "failed",
                "stage": "train",
                "returncode": int(e.returncode),
                "test_pearson": None,
                "test_spearman": None,
            },
            out_dir / "gkmsvm_optional.metrics.json",
        )
        print("WARN: gkmtrain failed; skipping gkm-SVM baseline.")
        return

    fasta_test = out_dir / "test.fa"
    write_fasta(test, fasta_test)

    # Resolve model file produced by gkmtrain
    model_file = Path(str(model_prefix) + ".model.txt")
    if not model_file.exists():
        # Fallback: pick the first matching model file if naming differs
        cand = sorted(out_dir.glob(model_prefix.name + "*.model*.txt"))
        if cand:
            model_file = cand[0]

    out_score = out_dir / "test.scores.txt"
    # Some LS-GKM builds accept the *prefix*, others require the explicit *.model.txt.
    # Try model_file first (most common), then fallback to prefix if needed.
    tried = []
    ok = False
    for model_arg in [str(model_file), str(model_prefix)]:
        cmd_pred = [args.gkmpredict, str(fasta_test), model_arg, str(out_score)]
        tried.append(cmd_pred)
        print("Running:", " ".join(cmd_pred))
        try:
            subprocess.check_call(cmd_pred)
            ok = True
            break
        except subprocess.CalledProcessError:
            continue

    if not ok:
        write_json(
            {
                "model": "gkmsvm_optional",
                "status": "failed",
                "stage": "predict",
                "model_file": str(model_file),
                "returncode": 1,
                "tried": [" ".join(c) for c in tried],
                "test_pearson": None,
                "test_spearman": None,
            },
            out_dir / "gkmsvm_optional.metrics.json",
        )
        print("WARN: gkmpredict failed (both model file and prefix). Skipping gkm-SVM baseline.")
        return

    pred = _read_scores(out_score)
    y_test = pd.to_numeric(test["delta"], errors="coerce").values.astype(float)

    m = compute_metrics(y_test, pred)
    metrics = {
        "model": "gkmsvm_optional",
        "target": "delta",
        "q": float(args.q),
        "test_pearson": float(m["pearson"]),
        "test_spearman": float(m["spearman"]),
        "n_test": int(len(test)),
    }
    write_json(metrics, out_dir / "gkmsvm_optional.metrics.json")

    pred_df = test[["element_id"]].copy()
    pred_df["y_delta"] = y_test
    pred_df["pred_delta"] = pred
    pred_df.to_csv(out_dir / "gkmsvm_optional.test_predictions.tsv", sep="\t", index=False)

    print("DONE")
    print(metrics)


if __name__ == "__main__":
    main()
