#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

MODELS=(
  cnn3head_kmer_fused_ens
  cnn_delta_ens
  cnn_msres_wt_mt_delta3head_ens
  gkmsvm_optional
  kmer_delta_ens
  kmer_elasticnet_delta_ens
  nt_transformer_delta_ens
  kmer_nystroem_ridge_delta_ens
  onehot_ridge_delta_ens
)

TS="$(date +%Y%m%d_%H%M%S)"
PKG="mpra_maintext9_research_pkg_${TS}"
rm -rf "$PKG"
mkdir -p "$PKG"/{00_manifest,01_final_outputs,02_main_benchmark,03_external_benchmark,04_failure_modes,05_negative_controls,06_transfer,07_scripts,08_reference_pack}

LATEST_PACK="$(ls -1t review_pack_*.tar.gz 2>/dev/null | head -n1 || true)"
if [[ -z "$LATEST_PACK" ]]; then
  echo "ERROR: no review_pack_*.tar.gz found" >&2
  exit 1
fi

printf "%s\n" "${MODELS[@]}" > "$PKG/00_manifest/models_kept_maintext9.txt"
cat > "$PKG/00_manifest/README_maintext9_research_pkg.md" <<EOF
# MPRA main-text 9-model research package (v2)

Created: $(date)
Project root: $ROOT
Latest review pack used as reference: $LATEST_PACK

This package is a curated, self-contained result bundle centered on these 9 models:
$(printf -- '- %s\n' "${MODELS[@]}")

Goals:
1. retain manuscript-ready summary tables;
2. retain all key benchmark result tables for the 9 models;
3. retain per-model prediction/result directories needed for re-mining;
4. retain failure modes, negative controls, transfer and trim-robustness artifacts;
5. retain the latest review pack for provenance.
EOF

cat > "$PKG/00_manifest/MANUSCRIPT_USE_SUMMARY.md" <<'EOF'
Use these files first when writing:
- 01_final_outputs/00_MASTER_SUMMARY.md
- 01_final_outputs/01_ceiling_framework/ceiling_benchmark_best_points.tsv
- 01_final_outputs/02_gap_decomposition/ceiling_gap_R.table.tsv
- 01_final_outputs/05_transfer/transfer_matrix.tsv
- 01_final_outputs/14_negative_controls/label_shuffle_summary.tsv
- 01_final_outputs/17_trim_robustness/trim_robustness_summary.tsv
- 02_main_benchmark/bootstrap_ci_report.maintext9.tsv
- 03_external_benchmark/gse142696_plan8.paper_table_by_trim.maintext9.tsv
- 03_external_benchmark/seven_primary_tasks_R_only_ranking.maintext9.tsv
- 03_external_benchmark/seven_primary_tasks_R_only_summary.maintext9.tsv
EOF

for d in 00_ceiling_ci 00_MASTER_SUMMARY.md 01_ceiling_framework 02_gap_decomposition 05_transfer 13_model_families 14_negative_controls 17_trim_robustness 25_endpoint_spec; do
  if [[ -e "nm_analysis_outputs_v4_final/$d" ]]; then
    cp -a "nm_analysis_outputs_v4_final/$d" "$PKG/01_final_outputs/"
  fi
done

python - "$ROOT" "$PKG" <<'PY'
from pathlib import Path
import pandas as pd
import sys
root=Path(sys.argv[1]); pkg=Path(sys.argv[2])
models=[
  'cnn3head_kmer_fused_ens','cnn_delta_ens','cnn_msres_wt_mt_delta3head_ens','gkmsvm_optional',
  'kmer_delta_ens','kmer_elasticnet_delta_ens','nt_transformer_delta_ens',
  'kmer_nystroem_ridge_delta_ens','onehot_ridge_delta_ens'
]

# main benchmark filtered leaderboard
src=root/'out_plan8'/'summary_vnext'/'bootstrap_ci_report.tsv'
outdir=pkg/'02_main_benchmark'; outdir.mkdir(parents=True, exist_ok=True)
if src.exists():
    df=pd.read_csv(src, sep='\t')
    mcol='model' if 'model' in df.columns else df.columns[0]
    dff=df[df[mcol].isin(models)].copy()
    dff.to_csv(outdir/'bootstrap_ci_report.maintext9.tsv', sep='\t', index=False)
    if {'stratum', mcol, 'pearson'}.issubset(df.columns):
        r=dff[dff['stratum'].astype(str).eq('R')].sort_values('pearson', ascending=False)
        r.to_csv(outdir/'R_only_ranking.maintext9.tsv', sep='\t', index=False)

# external paper table filtered
src=root/'out_gse142696_plan8'/'design_summary'/'gse142696_plan8.paper_table_by_trim.tsv'
outdir=pkg/'03_external_benchmark'; outdir.mkdir(parents=True, exist_ok=True)
if src.exists():
    df=pd.read_csv(src, sep='\t')
    mcol='model' if 'model' in df.columns else df.columns[0]
    dff=df[df[mcol].isin(models)].copy()
    dff.to_csv(outdir/'gse142696_plan8.paper_table_by_trim.maintext9.tsv', sep='\t', index=False)
    if {'design','trim',mcol}.issubset(dff.columns):
        # create an explicit R-only ranking table regardless of original schema
        if 'pearson_R' in dff.columns:
            rr=dff.sort_values(['design','trim','pearson_R'], ascending=[True,True,False])
            rr.to_csv(outdir/'R_only_ranking.by_design_trim.maintext9.tsv', sep='\t', index=False)
        elif {'stratum','pearson'}.issubset(dff.columns):
            rr=dff[dff['stratum'].astype(str).eq('R')].sort_values(['design','trim','pearson'], ascending=[True,True,False])
            rr.to_csv(outdir/'R_only_ranking.by_design_trim.maintext9.tsv', sep='\t', index=False)

# seven primary tasks ranking
rows=[]
main_src=root/'out_plan8'/'summary_vnext'/'bootstrap_ci_report.tsv'
if main_src.exists():
    df=pd.read_csv(main_src, sep='\t')
    mcol='model' if 'model' in df.columns else df.columns[0]
    if {'stratum','pearson'}.issubset(df.columns):
        x=df[(df[mcol].isin(models)) & (df['stratum'].astype(str)=='R')].copy()
        x['task']='GSE83894_main'
        rows.append(x[[mcol,'task','pearson']].rename(columns={mcol:'model'}))

ext_src=root/'out_gse142696_plan8'/'design_summary'/'gse142696_plan8.paper_table_by_trim.tsv'
if ext_src.exists():
    df=pd.read_csv(ext_src, sep='\t')
    mcol='model' if 'model' in df.columns else df.columns[0]
    x=df[df[mcol].isin(models)].copy()
    if 'pearson_R' in x.columns:
        x=x[['design','trim',mcol,'pearson_R']].rename(columns={mcol:'model','pearson_R':'pearson'})
    elif {'stratum','pearson'}.issubset(x.columns):
        x=x[x['stratum'].astype(str)=='R'][['design','trim',mcol,'pearson']].rename(columns={mcol:'model'})
    else:
        x=pd.DataFrame()
    if not x.empty:
        x['trim_label']=x['trim'].astype(float).astype(int).astype(str)
        x['task']=x['design'].astype(str)+'_trim'+x['trim_label']
        rows.append(x[['model','task','pearson']])

if rows:
    allr=pd.concat(rows, ignore_index=True)
    # drop duplicates just in case an upstream table has redundant rows
    allr=allr.drop_duplicates(subset=['model','task'])
    allr=allr.sort_values(['task','pearson'], ascending=[True,False])
    allr.to_csv(outdir/'seven_primary_tasks_R_only_ranking.maintext9.tsv', sep='\t', index=False)
    summary=[]
    for task, sub in allr.groupby('task', sort=True):
        sub=sub.reset_index(drop=True)
        sub['rank']=range(1,len(sub)+1)
        for _, r in sub.iterrows():
            summary.append({'task':task,'model':r['model'],'rank':int(r['rank']),'pearson':r['pearson']})
    s=pd.DataFrame(summary)
    agg=s.groupby('model').agg(
        n_tasks=('task','nunique'),
        mean_rank=('rank','mean'),
        wins_top1=('rank', lambda x: int((x==1).sum())),
        wins_top2=('rank', lambda x: int((x<=2).sum())),
        wins_top3=('rank', lambda x: int((x<=3).sum())),
        mean_pearson=('pearson','mean')
    ).reset_index().sort_values(['mean_rank','mean_pearson'], ascending=[True,False])
    agg.to_csv(outdir/'seven_primary_tasks_R_only_summary.maintext9.tsv', sep='\t', index=False)
PY

for m in "${MODELS[@]}"; do
  if [[ -d "out_plan8/results/split_seed0/ensemble/$m" ]]; then
    mkdir -p "$PKG/02_main_benchmark/results/split_seed0/ensemble"
    cp -a "out_plan8/results/split_seed0/ensemble/$m" "$PKG/02_main_benchmark/results/split_seed0/ensemble/"
  fi
  if [[ -d "posthoc_natmethods_analysis/failure_modes/main/$m" ]]; then
    mkdir -p "$PKG/04_failure_modes/main"
    cp -a "posthoc_natmethods_analysis/failure_modes/main/$m" "$PKG/04_failure_modes/main/"
  fi
done

for design in 3p3p 5p3p 5p5p; do
  for trim in 171 185; do
    base="out_gse142696_plan8/${design}/out_plan8_trim${trim}"
    if [[ -d "$base/summary_vnext" ]]; then
      mkdir -p "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}"
      cp -a "$base/summary_vnext" "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}/"
    fi
    if [[ -d "$base/summary_plan8" ]]; then
      mkdir -p "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}"
      cp -a "$base/summary_plan8" "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}/"
    fi
    for m in "${MODELS[@]}"; do
      if [[ -d "$base/results/split_seed0/ensemble/$m" ]]; then
        mkdir -p "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}/results/split_seed0/ensemble"
        cp -a "$base/results/split_seed0/ensemble/$m" "$PKG/03_external_benchmark/${design}/out_plan8_trim${trim}/results/split_seed0/ensemble/"
      fi
      if [[ -d "posthoc_natmethods_analysis/failure_modes/external/${design}/trim${trim}/${m}" ]]; then
        mkdir -p "$PKG/04_failure_modes/external/${design}/trim${trim}"
        cp -a "posthoc_natmethods_analysis/failure_modes/external/${design}/trim${trim}/${m}" "$PKG/04_failure_modes/external/${design}/trim${trim}/"
      fi
    done
  done
done

if [[ -d posthoc_natcomm_plus/negctrl ]]; then
  cp -a posthoc_natcomm_plus/negctrl "$PKG/05_negative_controls/"
fi

for d in posthoc_natmethods_analysis/transfer_R posthoc_natmethods_analysis/transfer_R_full posthoc_natmethods_analysis/failure_modes; do
  if [[ -d "$d" ]]; then
    cp -a "$d" "$PKG/06_transfer/"
  fi
done

cp -a "$LATEST_PACK" "$PKG/08_reference_pack/"

SCRIPT_KEEP=(
  scripts/summarize_gse142696_paper_table.py
  scripts/summarize_ci_plan8.py
  scripts/ensemble_predictions.py
  scripts/fuse_two_models_linear.py
  scripts/train_cnn_single.py
  scripts/train_cnn_wt_mt_delta3head_msres.py
  scripts/baseline_kmer_ridge.py
  scripts/baseline_kmer_sgd_elasticnet.py
  scripts/baseline_gkmsvm_optional.py
  scripts/baseline_kmer_nystroem_ridge.py
  scripts/baseline_onehot_ridge.py
  scripts/train_transformer_delta.py
  run_plan8_all_parallel.sh
  run_plan8_gse142696_parallel.sh
  run_plan8_gse142696_strong_models.sh
  run_plan8_failure_modes_plus.sh
  run_plan8_transfer_matrix_R_only.sh
  9.sh
  10_natmethods_analysis_upgrade.sh
  apply_external13_model_patches.sh
  rerun_external13_comparison.sh
  fix_trim_robustness_paper_table_R.sh
)
for f in "${SCRIPT_KEEP[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "$PKG/07_scripts/$(dirname "$f")"
    cp -a "$f" "$PKG/07_scripts/$f"
  fi
done

{
  echo "PACKAGE_DIR=$PKG"
  echo "LATEST_REVIEW_PACK=$LATEST_PACK"
  echo "MODELS=${MODELS[*]}"
  echo ""
  echo "Copied main benchmark result dirs:"; find "$PKG/02_main_benchmark" -maxdepth 4 -type d | sort
  echo ""
  echo "Copied external benchmark result dirs:"; find "$PKG/03_external_benchmark" -maxdepth 5 -type d | sort
} > "$PKG/00_manifest/package_manifest.txt"

tar -czf "$PKG.tar.gz" "$PKG"

echo "DONE"
echo "PACKAGE_DIR=$PKG"
echo "PACKAGE_TAR=$PKG.tar.gz"
