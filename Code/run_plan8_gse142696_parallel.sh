#!/usr/bin/env bash
set -euo pipefail

# Plan8 runner for GEO GSE142696 (9MPRA) with designs: 3p3p, 5p3p, 5p5p
# Builds UniReg-style exports and runs Plan8 per design and trim.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/scripts:${PYTHONPATH:-}"

PYBIN="$(command -v python)"
export PYBIN
echo "[env] PYBIN=$PYBIN" >&2


# --- dependency check (CPU PyTorch required even on CPU nodes) ---
$PYBIN -c "import torch" >/dev/null 2>&1 || {
  echo "ERROR: Python cannot import torch (PyTorch)." >&2
  echo "Your environment is missing PyTorch. Install CPU PyTorch, then re-run:" >&2
  echo "  conda install -c pytorch pytorch cpuonly -y" >&2
  echo "or (pip CPU wheels):" >&2
  echo "  $PYBIN -m pip install --index-url https://download.pytorch.org/whl/cpu torch" >&2
  exit 1
}
# --------------------------------------------------------------


ELEMENTS_FA=""
MEAN_TSV=""
REPS_TSV=""
PAIRS="5p3p,5p5p,3p3p"
TRIM_TO_LIST="171,185"
OUT_ROOT=""
SPLIT_SEEDS="0,1,2,3,4"
MODEL_SEEDS="0,1,2,3,4"
LOSS="huber"
DELTA_CLIP_Q="0.01"
DROP_BAD_REPS="yes"   # yes/no
RUN_GKMSVM="auto"     # auto|yes|no
N_JOBS="${N_JOBS:-4}"
BUILD_PAPER_TABLE="no"  # yes/no (avoid warnings when running a single design)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --elements_fa) ELEMENTS_FA="$2"; shift 2;;
    --activity_mean_tsv) MEAN_TSV="$2"; shift 2;;
    --activity_reps_tsv) REPS_TSV="$2"; shift 2;;
    --pairs) PAIRS="$2"; shift 2;;
    --trim_to_list) TRIM_TO_LIST="$2"; shift 2;;
    --out_root) OUT_ROOT="$2"; shift 2;;
    --split_seeds) SPLIT_SEEDS="$2"; shift 2;;
    --model_seeds) MODEL_SEEDS="$2"; shift 2;;
    --loss) LOSS="$2"; shift 2;;
    --delta_clip_q) DELTA_CLIP_Q="$2"; shift 2;;
    --drop_bad_reps) DROP_BAD_REPS="$2"; shift 2;;
    --run_gkmsvm) RUN_GKMSVM="$2"; shift 2;;
    --jobs) N_JOBS="$2"; shift 2;;
    --build_paper_table) BUILD_PAPER_TABLE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
 done

if [[ -z "$ELEMENTS_FA" || -z "$MEAN_TSV" || -z "$REPS_TSV" || -z "$OUT_ROOT" ]]; then
  echo "Missing required args: --elements_fa --activity_mean_tsv --activity_reps_tsv --out_root";
  exit 1
fi

mkdir -p "$OUT_ROOT"

IFS=',' read -ra PA <<< "$PAIRS"
IFS=',' read -ra TL <<< "$TRIM_TO_LIST"

for P in "${PA[@]}"; do
  echo "==== [PAIR] $P ===="
  PAIR_DIR="$OUT_ROOT/$P"
  mkdir -p "$PAIR_DIR/exports"

  DROP_FLAG="--drop_bad_reps"
  if [[ "$DROP_BAD_REPS" == "no" ]]; then
    DROP_FLAG="--keep_bad_reps"
  fi

  $PYBIN "$SCRIPT_DIR/scripts/gse142696_build_exports.py" \
    --elements_fa "$ELEMENTS_FA" \
    --activity_mean_tsv "$MEAN_TSV" \
    --activity_reps_tsv "$REPS_TSV" \
    --pair "$P" \
    --out_dir "$PAIR_DIR/exports" \
    $DROP_FLAG

  FORMATB="$PAIR_DIR/exports/formatB_agg_only.zip"
  FORMATA="$PAIR_DIR/exports/formatA_all_replicates.zip"

  for TRIM in "${TL[@]}"; do
    TAG="trim${TRIM}"
    echo "---- [RUN] $P / $TAG ----"

    bash "$SCRIPT_DIR/run_plan8_all_parallel.sh" \
      --formatB_zip "$FORMATB" \
      --formatA_zip "$FORMATA" \
      --out_dir "$PAIR_DIR/out_plan8_$TAG" \
      --split_seeds "$SPLIT_SEEDS" \
      --model_seeds "$MODEL_SEEDS" \
      --auto_trim yes \
      --trim_to "$TRIM" \
      --loss "$LOSS" \
      --delta_clip_q "$DELTA_CLIP_Q" \
      --run_gkmsvm "$RUN_GKMSVM" \
      --jobs "$N_JOBS"

    # design-specific gap analysis quick report (uses CI output)
    $PYBIN "$SCRIPT_DIR/scripts/design_gap_analysis.py" \
      --design_root "$PAIR_DIR/out_plan8_$TAG" \
      --out_md "$PAIR_DIR/out_plan8_$TAG/summary_plan8/design_gap_analysis.md" || true
  done
 done

if [[ "$BUILD_PAPER_TABLE" == "yes" ]]; then
  # Across designs: build paper table
  mkdir -p "$OUT_ROOT/design_summary"
  $PYBIN "$SCRIPT_DIR/scripts/summarize_gse142696_paper_table.py" \
    --root "$OUT_ROOT" \
    --out_tsv "$OUT_ROOT/design_summary/gse142696_plan8.paper_table_by_trim.tsv"
  # Backward-compat alias (keeps the old filename around)
  cp -f "$OUT_ROOT/design_summary/gse142696_plan8.paper_table_by_trim.tsv" "$OUT_ROOT/design_summary/gse142696_plan8.paper_table.tsv"
else
  echo "[INFO] Skip cross-design paper table (BUILD_PAPER_TABLE=$BUILD_PAPER_TABLE). Run: bash 2.sh or set --build_paper_table yes" >&2
fi

echo "DONE GSE142696 Plan8. Root: $OUT_ROOT"
