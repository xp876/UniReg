#!/usr/bin/env bash
set -euo pipefail

# GSE142696 external validation (plan8). Two options:
#  - Recommended (parallel): run the three designs concurrently on one node:
#       JOBS_PER_DESIGN=4 bash 2-parallel.sh
#  - Single command (one internal pool across all designs):
#       bash 2.sh

export MPLBACKEND=${MPLBACKEND:-Agg}
unset DISPLAY || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ELEMENTS_FA=${ELEMENTS_FA:-./GSM4237954_9MPRA_elements.fa.gz}
MEAN_TSV=${MEAN_TSV:-./GSE142696_9MPRA.ActivityRatios.tsv.gz}
REPS_TSV=${REPS_TSV:-./GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz}
OUT_ROOT=${OUT_ROOT:-./out_gse142696_plan8}

# If PARALLEL_DESIGNS=1, dispatch 2-1/2-2/2-3 in parallel.
PARALLEL_DESIGNS=${PARALLEL_DESIGNS:-0}

if [[ "$PARALLEL_DESIGNS" == "1" ]]; then
  JOBS_PER_DESIGN=${JOBS_PER_DESIGN:-4}
  export ELEMENTS_FA MEAN_TSV REPS_TSV OUT_ROOT JOBS_PER_DESIGN
  bash "$SCRIPT_DIR/2-parallel.sh"
  exit 0
fi

bash "$SCRIPT_DIR/run_plan8_gse142696_single_node.sh" \
  --elements_fa "$ELEMENTS_FA" \
  --activity_mean_tsv "$MEAN_TSV" \
  --activity_reps_tsv "$REPS_TSV" \
  --out_root "$OUT_ROOT" \
  --pairs 3p3p,5p3p,5p5p \
  --trim_to_list 171,185 \
  --split_seeds 0,1,2,3,4 \
  --model_seeds 0,1,2,3,4 \
  --loss huber \
  --delta_clip_q 0.01 \
  --jobs ${JOBS:-8} \
  --build_paper_table yes
