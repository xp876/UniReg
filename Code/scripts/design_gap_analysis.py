import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def read_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_one(root: Path, name: str) -> Optional[Path]:
    p = root / name
    if p.exists():
        return p
    for cand in root.rglob(name):
        return cand
    return None


def ceiling_R(ceiling: Dict) -> float:
    if not ceiling:
        return float("nan")
    if "DELTA" in ceiling and "by_element_prefix" in ceiling["DELTA"]:
        return float(ceiling["DELTA"]["by_element_prefix"].get("R", {}).get("pairwise_mean_pearson", np.nan))
    if "DELTA" in ceiling and "pairwise_mean_pearson" in ceiling["DELTA"]:
        return float(ceiling["DELTA"].get("pairwise_mean_pearson", np.nan))
    if "delta" in ceiling and "mean_pairwise_pearson" in ceiling["delta"]:
        return float(ceiling["delta"].get("mean_pairwise_pearson", np.nan))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_root", required=True, help="single design folder (e.g., out_gse142696_plan7/3p3p)")
    ap.add_argument("--ci_tsv", default="", help="bootstrap_ci_report.tsv (optional)")
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    droot = Path(args.design_root)

    ci_path = Path(args.ci_tsv) if args.ci_tsv else None
    if ci_path is None or not ci_path.exists():
        ci_path = find_one(droot, "bootstrap_ci_report.tsv")

    ceiling_path = find_one(droot, "ceiling_formatA_gb_by_prefix.json") or find_one(droot, "ceiling_formatA_gb.json")
    ceiling = read_json(ceiling_path) if ceiling_path else {}

    # target diagnostics (if available)
    # prefer analysis/split_seed0/targets_test.json (from analyze_targets.py)
    targ_path = find_one(droot, "targets_test.json")
    targ = read_json(targ_path) if targ_path else {}

    # parse CI
    ci = pd.read_csv(ci_path, sep="\t") if ci_path else pd.DataFrame()

    def get(ci_df, model, stratum):
        sub = ci_df[(ci_df["model"] == model) & (ci_df["stratum"] == stratum)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return float(r["pearson"]), float(r["ci95_lo"]), float(r["ci95_hi"]), int(r["n_total"])

    main_model = "cnn_wt_mt_delta_ens"
    kmer = "kmer_delta_ens"

    r_main = get(ci, main_model, "R")
    r_kmer = get(ci, kmer, "R")
    o_main = get(ci, main_model, "overall")
    o_kmer = get(ci, kmer, "overall")

    ceil_r = ceiling_R(ceiling)

    lines = []
    lines.append(f"# Design gap analysis: {droot.name}\n")

    lines.append("## Key takeaways (auto-generated)\n")
    if r_main:
        ratio = r_main[0] / ceil_r if np.isfinite(ceil_r) and ceil_r > 1e-9 else float("nan")
        lines.append(f"- **R-only Δ Pearson (CNN)**: {r_main[0]:.3f} [{r_main[1]:.3f}, {r_main[2]:.3f}] (n={r_main[3]})")
        if np.isfinite(ceil_r):
            lines.append(f"- **R-only ceiling** (replicate reliability): {ceil_r:.3f} → **CNN/ceiling** ≈ {ratio:.2f}")
    if r_kmer:
        lines.append(f"- **R-only Δ Pearson (k-mer ridge)**: {r_kmer[0]:.3f} [{r_kmer[1]:.3f}, {r_kmer[2]:.3f}] (n={r_kmer[3]})")

    lines.append("\n## Hypotheses to explain ceiling gap (what to test next)\n")
    lines.append("1) **Label noise / replicate inconsistency dominates**: if ceiling is low, models cannot exceed it.\n")
    lines.append("2) **Delta dynamic range differs by design**: small |delta| range makes ranking/regression harder.\n")
    lines.append("3) **Sequence grammar complexity differs**: 5p5p may require higher-order interactions beyond local motifs.\n")
    lines.append("4) **Outliers / heavy tails**: robust loss + winsorization should help if present.\n")

    lines.append("\n## Quick diagnostics (if available)\n")
    if targ:
        # expected keys from analyze_targets.py: delta_stats, prefix_stats etc; keep robust
        if "delta" in targ and isinstance(targ["delta"], dict):
            ds = targ["delta"]
            lines.append(f"- delta mean={ds.get('mean')}, std={ds.get('std')}, min={ds.get('min')}, max={ds.get('max')}")
        if "by_prefix" in targ and isinstance(targ["by_prefix"], dict) and "R" in targ["by_prefix"]:
            rp = targ["by_prefix"]["R"].get("delta", {})
            if isinstance(rp, dict) and rp:
                lines.append(f"- R-only delta std={rp.get('std')}, range=[{rp.get('min')},{rp.get('max')}]")

    lines.append("\n## What to run (concrete)\n")
    lines.append("- Learning curve on R-only: 20/40/60/80/100% training sizes (CNN + k-mer).")
    lines.append("- Robust delta loss sweep: huber_beta {0.25,0.5,1.0} × delta_clip_q {0,0.01,0.02}.")
    lines.append("- Check residuals: plot y_delta vs pred_delta + outlier list for top |residual| in R.")
    lines.append("- Compare motif match counts in conv1 between 3p3p and 5p5p: if 5p5p needs grammar, expect less single-motif dominance.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", args.out_md)


if __name__ == "__main__":
    main()
