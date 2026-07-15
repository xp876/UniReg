import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common import write_json


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 8:
        return float("nan")
    return float(np.corrcoef(y[m], p[m])[0, 1])


def load_pred(path: Path, y_col: str = "y_delta", p_col: str = "pred_delta") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "element_id" not in df.columns:
        raise SystemExit(f"Missing element_id in {path}")
    if y_col not in df.columns or p_col not in df.columns:
        raise SystemExit(f"Missing {y_col}/{p_col} in {path}")
    df = df[["element_id", y_col, p_col]].copy()
    df["prefix"] = df["element_id"].astype(str).map(element_prefix)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df.dropna(subset=[y_col, p_col]).copy()
    df = df.rename(columns={y_col: "y", p_col: "p"})
    return df


def boot_split_mean(
    by_split: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    splits = sorted(by_split.keys())
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals = []
        for s in splits:
            y, p = by_split[s]
            n = len(y)
            if n < 8:
                continue
            idx = rng.integers(0, n, size=n)
            vals.append(pearson(y[idx], p[idx]))
        out[i] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def ci95(samples: np.ndarray) -> Tuple[float, float]:
    return (float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975)))


def collect_preds(root: Path, pattern: str) -> Dict[str, Path]:
    """Return mapping split_seedX -> pred_tsv path for a given model pattern."""
    out = {}
    for p in root.rglob(pattern):
        m = re.search(r"split_seed\d+", str(p))
        if not m:
            continue
        out[m.group(0)] = p
    return out


def subset_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "overall":
        return df
    if mode == "non_control":
        return df[df["prefix"] != "C"].copy()
    if mode in {"R", "A", "C"}:
        return df[df["prefix"] == mode].copy()
    raise ValueError(f"Unknown mode: {mode}")


def summarize_model(root: Path, model_name: str, pred_paths: Dict[str, Path], n_boot: int, seed: int) -> Dict:
    # build split arrays for each stratum
    strata = ["overall", "non_control", "R", "A", "C"]
    out = {"model": model_name, "splits": sorted(pred_paths.keys()), "by_stratum": {}}

    for st in strata:
        by_split = {}
        n_total = 0
        for sp, path in pred_paths.items():
            df = load_pred(path)
            df = subset_df(df, st)
            y = df["y"].values.astype(float)
            p = df["p"].values.astype(float)
            by_split[sp] = (y, p)
            n_total += len(y)

        point = float(np.nanmean([pearson(y, p) for (y, p) in by_split.values() if len(y) >= 8]))
        boots = boot_split_mean(by_split, n_boot=n_boot, seed=seed + hash((model_name, st)) % 100000)
        out["by_stratum"][st] = {
            "n_total": int(n_total),
            "pearson_point": point,
            "pearson_ci95": ci95(boots),
        }
    return out


def paired_diff(root: Path, predA: Dict[str, Path], predB: Dict[str, Path], st: str, n_boot: int, seed: int) -> Dict:
    splits = sorted(set(predA.keys()).intersection(set(predB.keys())))
    if not splits:
        return {"error": "no overlapping splits"}

    by_split = {}
    for sp in splits:
        dfA = subset_df(load_pred(predA[sp]), st)
        dfB = subset_df(load_pred(predB[sp]), st)
        m = pd.merge(dfA, dfB, on="element_id", suffixes=("_A", "_B"))
        y = m["y_A"].values.astype(float)
        pA = m["p_A"].values.astype(float)
        pB = m["p_B"].values.astype(float)
        by_split[sp] = (y, pA, pB)

    # point
    pts = []
    for sp, (y, pA, pB) in by_split.items():
        if len(y) < 8:
            continue
        pts.append(pearson(y, pA) - pearson(y, pB))
    point = float(np.nanmean(pts)) if pts else float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals = []
        for sp, (y, pA, pB) in by_split.items():
            n = len(y)
            if n < 8:
                continue
            idx = rng.integers(0, n, size=n)
            vals.append(pearson(y[idx], pA[idx]) - pearson(y[idx], pB[idx]))
        boots[i] = float(np.nanmean(vals)) if vals else float("nan")

    # one-sided p-value for H1: corr(A) > corr(B)
    p_le0 = float(np.nanmean(boots <= 0.0))

    return {
        "stratum": st,
        "splits": splits,
        "delta_pearson_point": point,
        "delta_pearson_ci95": ci95(boots),
        "p_value_one_sided_le0": p_le0,
    }


def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction.

    Returns q-values in the original order. NaNs are passed through.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    msk = np.isfinite(p)
    if not msk.any():
        return q.tolist()
    pv = p[msk]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0.0, 1.0)
    # map back
    tmp = np.empty_like(qv)
    tmp[order] = qv
    q[msk] = tmp
    return q.tolist()


def main():
    ap = argparse.ArgumentParser()
    # Preferred CLI
    ap.add_argument("--root", required=False, help="out_plan8 root directory")
    ap.add_argument("--out_dir", required=False, help="directory to write reports")
    # Backward-compatible alias used by earlier wrappers
    ap.add_argument("--out_json", required=False, help="legacy JSON output path; out_dir inferred from parent")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_json_legacy = Path(args.out_json) if args.out_json else None

    # Supported forms:
    #   New: --root <root> --out_dir <summary_dir> [--out_json <summary_dir/file.json>]
    #   Legacy: --out_dir <root> --out_json <summary_dir/file.json>
    if args.root:
        root = Path(args.root)
        out_dir = Path(args.out_dir) if args.out_dir else (out_json_legacy.parent if out_json_legacy else None)
    else:
        # Legacy wrapper used --out_dir as the root and --out_json as the actual file path
        if out_json_legacy is not None and args.out_dir is not None:
            root = Path(args.out_dir)
            out_dir = out_json_legacy.parent
        else:
            root = None
            out_dir = Path(args.out_dir) if args.out_dir else (out_json_legacy.parent if out_json_legacy else None)
            # If only a summary dir is given (no root), we cannot reliably infer root; keep None and error below.
    if root is None or out_dir is None:
        raise SystemExit("Need --root and --out_dir (or legacy --out_json with inferable parent)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # preferred prediction files (ensembles)
    models = {
        "cnn_wt_mt_delta_ens": collect_preds(root, "cnn_wt_mt_delta_ens.test_predictions.tsv"),
        "cnn_wt_mt_delta3head_ens": collect_preds(root, "cnn_wt_mt_delta3head_ens.test_predictions.tsv"),
        "cnn_msres_wt_mt_delta3head_ens": collect_preds(root, "cnn_msres_wt_mt_delta3head_ens.test_predictions.tsv"),
        "nt_transformer_delta_ens": collect_preds(root, "nt_transformer_delta_ens.test_predictions.tsv"),
        "cnn_wt_mt_derive_delta_ens": collect_preds(root, "cnn_wt_mt_derive_delta_ens.test_predictions.tsv"),
        "cnn_delta_ens": collect_preds(root, "cnn_delta_ens.test_predictions.tsv"),
        "kmer_delta_ens": collect_preds(root, "kmer_delta_ens.test_predictions.tsv"),
        "onehot_ridge_delta_ens": collect_preds(root, "onehot_ridge_delta_ens.test_predictions.tsv"),
        "kmer_elasticnet_delta_ens": collect_preds(root, "kmer_elasticnet_delta_ens.test_predictions.tsv"),
        "kmer_nystroem_ridge_delta_ens": collect_preds(root, "kmer_nystroem_ridge_delta_ens.test_predictions.tsv"),
        "cnn_kmer_fused_ens": collect_preds(root, "cnn_kmer_fused_ens.test_predictions.tsv"),
        "cnn3head_kmer_fused_ens": collect_preds(root, "cnn3head_kmer_fused_ens.test_predictions.tsv"),
        "gkmsvm_optional": collect_preds(root, "gkmsvm_optional.test_predictions.tsv"),
    }

    # filter empty
    models = {k: v for k, v in models.items() if v}
    if not models:
        raise SystemExit(f"No prediction files found under {root}")

    report = {"root": str(root), "n_boot": int(args.n_boot), "seed": int(args.seed), "models": [] , "paired": []}

    for name, paths in models.items():
        report["models"].append(summarize_model(root, name, paths, n_boot=args.n_boot, seed=args.seed))

    # paired comparisons for main story
    # compare improved 3-head model vs baseline CNN and k-mer
    if "cnn_wt_mt_delta3head_ens" in models and "cnn_wt_mt_delta_ens" in models:
        report["paired"].append({
            "A": "cnn_wt_mt_delta3head_ens",
            "B": "cnn_wt_mt_delta_ens",
            "overall": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["cnn_wt_mt_delta_ens"], "overall", args.n_boot, args.seed + 111),
            "R_only": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["cnn_wt_mt_delta_ens"], "R", args.n_boot, args.seed + 222),
        })
    if "cnn_wt_mt_delta3head_ens" in models and "kmer_delta_ens" in models:
        report["paired"].append({
            "A": "cnn_wt_mt_delta3head_ens",
            "B": "kmer_delta_ens",
            "overall": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["kmer_delta_ens"], "overall", args.n_boot, args.seed + 333),
            "R_only": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["kmer_delta_ens"], "R", args.n_boot, args.seed + 444),
        })

    # strong model comparisons (NatMethods Analysis upgrade)
    if "cnn_msres_wt_mt_delta3head_ens" in models and "cnn_wt_mt_delta3head_ens" in models:
        report["paired"].append({
            "A": "cnn_msres_wt_mt_delta3head_ens",
            "B": "cnn_wt_mt_delta3head_ens",
            "overall": paired_diff(root, models["cnn_msres_wt_mt_delta3head_ens"], models["cnn_wt_mt_delta3head_ens"], "overall", args.n_boot, args.seed + 555),
            "R_only": paired_diff(root, models["cnn_msres_wt_mt_delta3head_ens"], models["cnn_wt_mt_delta3head_ens"], "R", args.n_boot, args.seed + 666),
        })
    if "nt_transformer_delta_ens" in models and "kmer_delta_ens" in models:
        report["paired"].append({
            "A": "nt_transformer_delta_ens",
            "B": "kmer_delta_ens",
            "overall": paired_diff(root, models["nt_transformer_delta_ens"], models["kmer_delta_ens"], "overall", args.n_boot, args.seed + 777),
            "R_only": paired_diff(root, models["nt_transformer_delta_ens"], models["kmer_delta_ens"], "R", args.n_boot, args.seed + 888),
        })

    if "cnn_wt_mt_delta_ens" in models and "kmer_delta_ens" in models:
        report["paired"].append({
            "A": "cnn_wt_mt_delta_ens",
            "B": "kmer_delta_ens",
            "overall": paired_diff(root, models["cnn_wt_mt_delta_ens"], models["kmer_delta_ens"], "overall", args.n_boot, args.seed + 123),
            "R_only": paired_diff(root, models["cnn_wt_mt_delta_ens"], models["kmer_delta_ens"], "R", args.n_boot, args.seed + 456),
        })
    if "cnn_wt_mt_delta_ens" in models and "gkmsvm_optional" in models:
        report["paired"].append({
            "A": "cnn_wt_mt_delta_ens",
            "B": "gkmsvm_optional",
            "overall": paired_diff(root, models["cnn_wt_mt_delta_ens"], models["gkmsvm_optional"], "overall", args.n_boot, args.seed + 777),
            "R_only": paired_diff(root, models["cnn_wt_mt_delta_ens"], models["gkmsvm_optional"], "R", args.n_boot, args.seed + 888),
        })

    # fusion comparisons (should be reviewer-proof if improvement is consistent)
    if "cnn_kmer_fused_ens" in models and "cnn_wt_mt_delta_ens" in models:
        report["paired"].append({
            "A": "cnn_kmer_fused_ens",
            "B": "cnn_wt_mt_delta_ens",
            "overall": paired_diff(root, models["cnn_kmer_fused_ens"], models["cnn_wt_mt_delta_ens"], "overall", args.n_boot, args.seed + 1357),
            "R_only": paired_diff(root, models["cnn_kmer_fused_ens"], models["cnn_wt_mt_delta_ens"], "R", args.n_boot, args.seed + 2468),
        })
    if "cnn3head_kmer_fused_ens" in models and "cnn_wt_mt_delta3head_ens" in models:
        report["paired"].append({
            "A": "cnn3head_kmer_fused_ens",
            "B": "cnn_wt_mt_delta3head_ens",
            "overall": paired_diff(root, models["cnn3head_kmer_fused_ens"], models["cnn_wt_mt_delta3head_ens"], "overall", args.n_boot, args.seed + 3579),
            "R_only": paired_diff(root, models["cnn3head_kmer_fused_ens"], models["cnn_wt_mt_delta3head_ens"], "R", args.n_boot, args.seed + 4680),
        })

    # derived-Δ vs direct-Δ (mechanistic/"physical consistency" argument)
    if "cnn_wt_mt_derive_delta_ens" in models and "cnn_wt_mt_delta3head_ens" in models:
        report["paired"].append({
            "A": "cnn_wt_mt_delta3head_ens",
            "B": "cnn_wt_mt_derive_delta_ens",
            "overall": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["cnn_wt_mt_derive_delta_ens"], "overall", args.n_boot, args.seed + 5791),
            "R_only": paired_diff(root, models["cnn_wt_mt_delta3head_ens"], models["cnn_wt_mt_derive_delta_ens"], "R", args.n_boot, args.seed + 6802),
        })

    # attach BH-FDR q-values for p-values (overall/R_only separately)
    for key in ["overall", "R_only"]:
        pvals = []
        for comp in report["paired"]:
            d = comp.get(key, {})
            pvals.append(float(d.get("p_value_one_sided_le0", np.nan)))
        qvals = bh_fdr(pvals)
        for comp, q in zip(report["paired"], qvals):
            if key in comp and isinstance(comp[key], dict):
                comp[key]["q_value_bh_fdr"] = float(q) if np.isfinite(q) else float("nan")

    write_json(report, out_dir / "bootstrap_ci_report.json")
    if out_json_legacy is not None:
        out_json_legacy.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, out_json_legacy)

    # write a compact TSV (models)
    rows = []
    for m in report["models"]:
        for st, d in m["by_stratum"].items():
            lo, hi = d["pearson_ci95"]
            rows.append({
                "model": m["model"],
                "stratum": st,
                "pearson": d["pearson_point"],
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_total": d["n_total"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap_ci_report.tsv", sep="\t", index=False)

    # paired diff TSV (with p/q)
    prow = []
    for comp in report["paired"]:
        for key, st_name in [("overall", "overall"), ("R_only", "R")]:
            d = comp.get(key, {})
            if not isinstance(d, dict) or "delta_pearson_point" not in d:
                continue
            lo, hi = d.get("delta_pearson_ci95", (float("nan"), float("nan")))
            prow.append({
                "A": comp.get("A"),
                "B": comp.get("B"),
                "stratum": st_name,
                "delta_pearson": d.get("delta_pearson_point"),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "p_one_sided_le0": d.get("p_value_one_sided_le0"),
                "q_bh_fdr": d.get("q_value_bh_fdr"),
            })
    if prow:
        pd.DataFrame(prow).to_csv(out_dir / "paired_bootstrap_report.tsv", sep="\t", index=False)

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
