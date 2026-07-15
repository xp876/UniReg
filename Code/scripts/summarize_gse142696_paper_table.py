import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def read_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_ci_report(design_root: Path) -> Optional[Path]:
    """Find the Plan8 CI report within a run root.

    Preference order:
      1) summary_vnext/bootstrap_ci_report.tsv
      2) summary_plan8/bootstrap_ci_report.tsv
      3) first match anywhere under the root
    """
    c = design_root / "summary_vnext" / "bootstrap_ci_report.tsv"
    if c.exists():
        return c
    c = design_root / "summary_plan8" / "bootstrap_ci_report.tsv"
    if c.exists():
        return c
    for cand in design_root.rglob("bootstrap_ci_report.tsv"):
        return cand
    return None


def find_ceiling(design_root: Path) -> Optional[Dict]:
    """Find replicate ceiling json within a run root."""
    for name in [
        "ceiling_formatA_gb_by_prefix.json",
        "ceiling_formatA_gb.json",
        "ceiling_formatA.json",
    ]:
        p = design_root / name
        if p.exists():
            return read_json(p)

    # fallback: search recursively
    for p in design_root.rglob("ceiling_formatA_gb_by_prefix.json"):
        return read_json(p)
    for p in design_root.rglob("ceiling_formatA_gb.json"):
        return read_json(p)
    for p in design_root.rglob("ceiling_formatA.json"):
        return read_json(p)
    return None


def ceiling_delta_overall(ceiling: Dict) -> float:
    if not ceiling:
        return float("nan")
    if "DELTA" in ceiling and "pairwise_mean_pearson" in ceiling["DELTA"]:
        return float(ceiling["DELTA"].get("pairwise_mean_pearson", np.nan))
    if "delta" in ceiling and "mean_pairwise_pearson" in ceiling["delta"]:
        return float(ceiling["delta"].get("mean_pairwise_pearson", np.nan))
    return float("nan")


def ceiling_delta_by_prefix(ceiling: Dict, pref: str) -> float:
    """Return DELTA replicate ceiling for a given prefix (R/A/C).

    Supports both the new by-prefix schema and older overall-only schemas.
    """
    if not ceiling:
        return float("nan")

    # new schema
    if "DELTA" in ceiling and "by_element_prefix" in ceiling["DELTA"]:
        return float(
            ceiling["DELTA"]["by_element_prefix"].get(pref, {}).get("pairwise_mean_pearson", np.nan)
        )

    # old schema: overall only
    if "DELTA" in ceiling and "pairwise_mean_pearson" in ceiling["DELTA"]:
        return float(ceiling["DELTA"].get("pairwise_mean_pearson", np.nan))

    # sometimes "delta" key
    if "delta" in ceiling and "mean_pairwise_pearson" in ceiling["delta"]:
        return float(ceiling["delta"].get("mean_pairwise_pearson", np.nan))

    return float("nan")


def parse_trim(run_root: Path) -> float:
    m = re.search(r"trim(\d+)", run_root.name)
    return float(m.group(1)) if m else float("nan")


def iter_run_roots(design_root: Path):
    """Yield run roots for a GSE142696 design.

    Plan8 writes:
      <pair>/out_plan8_trim171/
      <pair>/out_plan8_trim185/

    If these exist, yield each. Otherwise yield the design_root itself.
    """
    subruns = sorted([p for p in design_root.glob("out_plan8_trim*") if p.is_dir()], key=lambda p: p.name)
    if subruns:
        for r in subruns:
            yield r
    else:
        yield design_root




def is_design_dir(p: Path) -> bool:
    """Heuristic filter for real GSE142696 design folders (e.g., 3p3p/5p3p/5p5p)."""
    if not p.is_dir():
        return False
    if p.name in {"design_summary", "logs", "summary_vnext", "summary_plan8", "__pycache__"}:
        return False
    # Typical layout: <design>/out_plan8_trim171 and /out_plan8_trim185
    if any(c.is_dir() and c.name.startswith("out_plan8_trim") for c in p.iterdir()):
        return True
    # Fallback: direct run root containing summary/ceiling outputs
    if (p / "summary_vnext").is_dir() or (p / "summary_plan8").is_dir():
        return True
    if any(p.glob("ceiling_formatA*.json")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    # Preferred interface
    ap.add_argument("--root", help="out_gse142696_plan8")
    # Backward-compatible aliases used by older wrapper scripts
    ap.add_argument("--ext_root", help="alias of --root", default=None)
    ap.add_argument("--main_out", default=None)
    ap.add_argument("--main_ci_json", default=None)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    root_arg = args.root if args.root else args.ext_root
    if not root_arg:
        raise SystemExit("Need --root (or legacy --ext_root)")
    root = Path(root_arg)
    designs = [p for p in root.iterdir() if is_design_dir(p)]
    designs = sorted(designs, key=lambda p: p.name)
    if not designs:
        raise SystemExit(f"No design subfolders found under {root}")

    rows = []
    keep_models = [
            "cnn_wt_mt_delta_ens",
            "cnn_wt_mt_delta3head_ens",
            "cnn_msres_wt_mt_delta3head_ens",
            "kmer_delta_ens",
            "cnn3head_kmer_fused_ens",
            "nt_transformer_delta_ens",
            "gkmsvm_optional",
            "cnn_kmer_fused_ens",
            "cnn_delta_ens",
            "cnn_wt_mt_derive_delta_ens",
            "onehot_ridge_delta_ens",
            "kmer_elasticnet_delta_ens",
            "kmer_nystroem_ridge_delta_ens",
        ]

    for droot in designs:
        for run_root in iter_run_roots(droot):
            ci_tsv = find_ci_report(run_root)
            if ci_tsv is None:
                print("WARN: no CI report found under run root:", run_root)
                continue

            ci = pd.read_csv(ci_tsv, sep="\t")
            ci = ci[ci["model"].isin(keep_models)].copy()

            ceiling = find_ceiling(run_root)
            ceil_overall = ceiling_delta_overall(ceiling)
            ceil_R = ceiling_delta_by_prefix(ceiling, "R")

            trim = parse_trim(run_root)

            for _, r in ci.iterrows():
                st = r["stratum"]
                pearson = float(r["pearson"])
                rows.append(
                    {
                        "design": droot.name,
                        "trim": trim,
                        "run_tag": run_root.name,
                        "run_dir": str(run_root),
                        "model": r["model"],
                        "stratum": st,
                        "pearson": pearson,
                        "ci95_lo": float(r["ci95_lo"]),
                        "ci95_hi": float(r["ci95_hi"]),
                        "n_total": int(r["n_total"]),
                        "ceiling_R": float(ceil_R),
                        "ceiling_overall": float(ceil_overall),
                        "pearson_over_ceiling_R": pearson / float(ceil_R)
                        if np.isfinite(ceil_R) and ceil_R > 1e-9 and st == "R"
                        else float("nan"),
                        "pearson_over_ceiling_overall": pearson / float(ceil_overall)
                        if np.isfinite(ceil_overall) and ceil_overall > 1e-9 and st == "overall"
                        else float("nan"),
                    }
                )

    out = pd.DataFrame(rows)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print("Wrote:", str(out_path))


if __name__ == "__main__":
    main()
