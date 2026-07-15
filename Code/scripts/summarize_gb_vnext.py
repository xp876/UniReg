
import argparse
from pathlib import Path
import json
import re
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_split(path: Path) -> str:
    s = str(path)
    m = re.search(r"split_seed\d+", s)
    return m.group(0) if m else "split_unknown"


def infer_seed(path: Path) -> int:
    s = str(path)
    m = re.search(r"/seed(\d+)/", s)
    return int(m.group(1)) if m else -1


def _extract_delta_pearson(d: Dict[str, Any]) -> float:
    # unify across models
    model = d.get("model")
    # Some *.metrics.json may omit 'model' (or it becomes NaN after DataFrame
    # construction). Guard against NaN / non-string values.
    if model is None:
        return float("nan")
    # covers numpy.nan, pandas.NA, etc.
    if pd.isna(model):
        return float("nan")
    if not isinstance(model, str):
        model = str(model)
    if model in ("cnn_single", "kmer_ridge") and d.get("target") == "delta":
        return float(d.get("test_pearson", np.nan))
    if model == "cnn_mean_delta":
        return float(d.get("test_delta_pearson", np.nan))
    if model == "cnn_wt_mt_delta":
        return float(d.get("test_delta_pearson", np.nan))
    if model == "cnn_wt_mt_delta3head":
        return float(d.get("test_delta_pearson", np.nan))
    if model.endswith("_ens"):
        # ensemble metrics stored as test_delta_pearson maybe
        for k in ["test_delta_pearson", "test_delta_pearson", "test_delta_pearson"]:
            if k in d:
                return float(d.get(k))
        # or test_delta_pearson in test_delta_pearson? fallback:
        for k in d.keys():
            if "delta_pearson" in k:
                return float(d[k])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="out directory")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load ceiling (optional)
    ceiling = None
    cpath = root / "ceiling_formatA_gb.json"
    if cpath.exists():
        ceiling = read_json(cpath)

    metric_files = list(root.rglob("*.metrics.json"))
    rows = []
    for mf in metric_files:
        d = read_json(mf)
        d["_path"] = str(mf)
        d["split"] = infer_split(mf)
        if "seed" not in d:
            d["seed"] = infer_seed(mf)
        rows.append(d)

    if not rows:
        raise SystemExit(f"No metrics found under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_metrics.tsv", sep="\t", index=False)

    # collect delta pearson per model
    df["delta_test_pearson"] = df.apply(lambda r: _extract_delta_pearson(r.to_dict()), axis=1)
    sub = df[np.isfinite(df["delta_test_pearson"])].copy()
    if not sub.empty:
        g = sub.groupby(["model"])["delta_test_pearson"].agg(["mean","std","count"]).reset_index()
        g = g.sort_values("mean", ascending=False)
        g.to_csv(out_dir / "delta_overall_model.tsv", sep="\t", index=False)

        # ceiling normalized
        if ceiling and "delta" in ceiling:
            delta_ceiling = float(ceiling["delta"].get("mean_pairwise_pearson", np.nan))
            if np.isfinite(delta_ceiling) and delta_ceiling > 1e-9:
                g["delta_ceiling"] = delta_ceiling
                g["delta_over_ceiling"] = g["mean"] / delta_ceiling
                g.to_csv(out_dir / "delta_overall_model_ceiling_norm.tsv", sep="\t", index=False)

        # plot
        plt.figure(figsize=(7,3))
        x = np.arange(len(g))
        plt.bar(x, g["mean"].values)
        plt.errorbar(x, g["mean"].values, yerr=g["std"].values, fmt="none", capsize=3)
        plt.xticks(x, g["model"].values, rotation=0)
        plt.ylabel("Test Pearson (delta)")
        plt.title("Delta predictability (overall)")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_delta_overall.png", dpi=200)
        plt.close()

    # stratified json aggregation
    strat_files = list(root.rglob("*.delta.stratified.json")) + list(root.rglob("*.stratified.vnext.json"))
    srows = []
    for sf in strat_files:
        d = read_json(sf)
        srows.append({
            "_path": str(sf),
            "split": infer_split(sf),
            "seed": infer_seed(sf),
            "overall_pearson": d.get("overall", {}).get("pearson"),
            "non_excl_pearson": d.get("overall_non_excluded", {}).get("pearson"),
            "R_pearson": (d.get("by_prefix", {}) or {}).get("R", {}).get("pearson"),
            "A_pearson": (d.get("by_prefix", {}) or {}).get("A", {}).get("pearson"),
            "C_pearson": (d.get("by_prefix", {}) or {}).get("C", {}).get("pearson"),
        })
    if srows:
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out_dir / "all_stratified_summaries.tsv", sep="\t", index=False)
        # aggregate by model inferred from path
        # aggregate by model inferred from path.
        # IMPORTANT: use exact directory-name matches to avoid mis-classifying
        # 'cnn_wt_mt_delta3head' as 'cnn_wt_mt_delta'.
        def infer_model(p: str) -> str:
            parts = Path(p).parts
            # Prefer longer, explicit names first.
            known = [
                "cnn_wt_mt_delta3head_ens",
                "cnn_wt_mt_delta3head",
                "cnn_wt_mt_delta_ens",
                "cnn_wt_mt_delta",
                "cnn_mean_delta",
                "cnn_delta_ens",
                "cnn_delta",
                "kmer_delta_ens",
                "kmer_delta",
                "kmer_mean",
                "dinuc_ridge",
            ]
            for name in known:
                if name in parts:
                    return name

            # Handle any ensemble_* folders explicitly.
            for part in parts:
                if part.startswith("ensemble_"):
                    return part

            # Fallback: parent dir name (usually the model folder)
            return Path(p).parent.name
        sdf["model"] = sdf["_path"].map(infer_model)
        g = sdf.groupby("model")[["overall_pearson","non_excl_pearson","R_pearson","A_pearson","C_pearson"]].agg(["mean","std","count"])
        g.to_csv(out_dir / "stratified_by_model.tsv", sep="\t")

    print("Wrote summaries to:", out_dir)

if __name__ == "__main__":
    main()
