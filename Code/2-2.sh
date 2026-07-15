#!/usr/bin/env bash
set -euo pipefail

# Run ONLY design 5p3p for GSE142696 (plan8).

export MPLBACKEND=${MPLBACKEND:-Agg}
unset DISPLAY || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ELEMENTS_FA=${ELEMENTS_FA:-./GSM4237954_9MPRA_elements.fa.gz}
MEAN_TSV=${MEAN_TSV:-./GSE142696_9MPRA.ActivityRatios.tsv.gz}
REPS_TSV=${REPS_TSV:-./GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz}
OUT_ROOT=${OUT_ROOT:-./out_gse142696_plan8}

JOBS_PER_DESIGN=${JOBS_PER_DESIGN:-4}

bash "$SCRIPT_DIR/run_plan8_gse142696_single_node.sh" \
  --elements_fa "$ELEMENTS_FA" \
  --activity_mean_tsv "$MEAN_TSV" \
  --activity_reps_tsv "$REPS_TSV" \
  --out_root "$OUT_ROOT" \
  --pairs 5p3p \
  --trim_to_list 171,185 \
  --split_seeds 0,1,2,3,4 \
  --model_seeds 0,1,2,3,4 \
  --loss huber \
  --delta_clip_q 0.01 \
  --jobs "$JOBS_PER_DESIGN" \
  --build_paper_table no
