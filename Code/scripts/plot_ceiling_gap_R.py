import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_one(root: Path, name: str) -> Path:
    cands = list(root.rglob(name))
    if not cands:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    # prefer shortest path (closest to root)
    cands = sorted(cands, key=lambda p: (len(p.parts), str(p)))
    return cands[0]


def load_ceiling_R(run_dir: Path, prefix: str = "R") -> float:
    js_path = None
    for cand in [
        "ceiling_formatA_gb_by_prefix.json",
        "ceiling_formatA_gb_by_prefix.WTMT.json",
        "ceiling_formatA_by_prefix.json",
    ]:
        cands = list(run_dir.rglob(cand))
        if cands:
            js_path = sorted(cands, key=lambda p: (len(p.parts), str(p)))[0]
            break
    if js_path is None:
        return float("nan")
    obj = json.loads(Path(js_path).read_text())
    try:
        return float(obj["DELTA"]["by_element_prefix"][prefix]["pairwise_mean_pearson"])
    except Exception:
        return float("nan")


def load_ci_row(ci_tsv: Path, model: str, stratum: str = "R") -> Dict:
    df = pd.read_csv(ci_tsv, sep="\t")
    sub = df[(df["model"] == model) & (df["stratum"] == stratum)]
    if len(sub) == 0:
        return {"pearson": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n_total": 0}
    r = sub.iloc[0].to_dict()
    return {
        "pearson": float(r.get("pearson", np.nan)),
        "ci95_lo": float(r.get("ci95_lo", np.nan)),
        "ci95_hi": float(r.get("ci95_hi", np.nan)),
        "n_total": int(r.get("n_total", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", required=True, help="Comma-separated run roots (e.g. out_plan8, out_gse142696_plan8/5p5p/out_plan8_trim185)")
    ap.add_argument("--labels", default="", help="Comma-separated labels; default uses basename of run dir")
    ap.add_argument("--models", default="cnn_wt_mt_delta_ens,kmer_delta_ens", help="Comma-separated model names as in bootstrap_ci_report.tsv")
    ap.add_argument("--ci_name", default="bootstrap_ci_report.tsv", help="Filename to search for")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="R", help="Element prefix for ceiling and CI strata")
    args = ap.parse_args()

    run_dirs = [Path(x.strip()) for x in args.run_dirs.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if labels and len(labels) != len(run_dirs):
        raise SystemExit("--labels must match number of --run_dirs")
    if not labels:
        labels = [d.name for d in run_dirs]

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir, lab in zip(run_dirs, labels):
        ceiling_r = load_ceiling_R(run_dir, prefix=args.prefix)

        # CI report: prefer summary_plan8 then summary_vnext
        ci_path = None
        for cand in [
            run_dir / "summary_plan8" / args.ci_name,
            run_dir / "summary_vnext" / args.ci_name,
        ]:
            if cand.exists():
                ci_path = cand
                break
        if ci_path is None:
            try:
                ci_path = find_one(run_dir, args.ci_name)
            except Exception:
                ci_path = None

        for model in models:
            if ci_path is None:
                r = {"pearson": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "n_total": 0}
            else:
                r = load_ci_row(ci_path, model=model, stratum="R" if args.prefix == "R" else args.prefix)

            rows.append({
                "run": lab,
                "run_dir": str(run_dir),
                "model": model,
                "pearson_R": r["pearson"],
                "ci95_lo": r["ci95_lo"],
                "ci95_hi": r["ci95_hi"],
                "n_R": r["n_total"],
                "ceiling_R": ceiling_r,
                "ceiling_norm": (r["pearson"] / ceiling_r) if (np.isfinite(r["pearson"]) and np.isfinite(ceiling_r) and ceiling_r != 0) else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ceiling_gap_R.table.tsv", sep="\t", index=False)

    # Plot: for each run, show ceiling + each model
    run_order = labels
    x = np.arange(len(run_order))

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(run_order), 4.2))

    # ceiling bars
    ceilings = [float(df[df["run"] == r]["ceiling_R"].iloc[0]) if len(df[df["run"] == r]) else np.nan for r in run_order]
    width = 0.75 / (len(models) + 1)
    ax.bar(x - 0.375 + width * 0.5, ceilings, width=width, label="replicate ceiling (R)")

    # model bars
    for i, model in enumerate(models):
        vals = []
        yerr_low = []
        yerr_high = []
        for r in run_order:
            sub = df[(df["run"] == r) & (df["model"] == model)]
            if len(sub) == 0:
                vals.append(np.nan); yerr_low.append(0.0); yerr_high.append(0.0)
                continue
            v = float(sub["pearson_R"].iloc[0])
            lo = float(sub["ci95_lo"].iloc[0])
            hi = float(sub["ci95_hi"].iloc[0])
            vals.append(v)
            yerr_low.append(max(0.0, v - lo) if np.isfinite(v) and np.isfinite(lo) else 0.0)
            yerr_high.append(max(0.0, hi - v) if np.isfinite(v) and np.isfinite(hi) else 0.0)
        pos = x - 0.375 + width * (i + 1.5)
        ax.bar(pos, vals, width=width, label=model, yerr=[yerr_low, yerr_high], capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(run_order, rotation=25, ha="right")
    ax.set_ylabel("Pearson r on Δ (R-only)")
    ax.set_title("Ceiling gap by design (R-only)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    fig.savefig(out_dir / "ceiling_gap_R.bar.png", dpi=200)
    fig.savefig(out_dir / "ceiling_gap_R.bar.pdf")

    # normalized plot
    fig2, ax2 = plt.subplots(figsize=(1.5 + 1.2 * len(run_order), 3.8))
    for i, model in enumerate(models):
        vals = []
        for r in run_order:
            sub = df[(df["run"] == r) & (df["model"] == model)]
            vals.append(float(sub["ceiling_norm"].iloc[0]) if len(sub) else np.nan)
        pos = x - 0.35 + (i + 0.5) * (0.7 / max(1, len(models)))
        ax2.bar(pos, vals, width=(0.7 / max(1, len(models))), label=model)

    ax2.axhline(1.0, linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_order, rotation=25, ha="right")
    ax2.set_ylabel("Ceiling-normalized r")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Ceiling-normalized performance (R-only)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "ceiling_gap_R.normalized.png", dpi=200)
    fig2.savefig(out_dir / "ceiling_gap_R.normalized.pdf")

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
