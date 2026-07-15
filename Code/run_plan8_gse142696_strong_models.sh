#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXT_ROOT="./out_gse142696_plan8"
PAIRS="3p3p,5p3p,5p5p"
TRIMS="171,185"
SPLIT_SEEDS="0,1,2,3,4"
MODEL_SEEDS="0,1,2,3,4"
JOBS="8"

ENABLE_TRANSFORMER="0"
TRANSFORMER_MODEL="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
TRANSFORMER_OFFLINE="0"
TRANSFORMER_CACHE_DIR=""
TRANSFORMER_REVISION=""
TRANSFORMER_MODE="cpu"
TRANSFORMER_DEVICE="cpu"
TRANSFORMER_FREEZE="yes"
TRANSFORMER_UNFREEZE_LAST_N="0"
TRANSFORMER_FP16="0"
TRANSFORMER_BATCH_SIZE="32"
TRANSFORMER_EPOCHS="40"
TRANSFORMER_PATIENCE="6"
TRANSFORMER_LR="3e-4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ext_root) EXT_ROOT="$2"; shift 2;;
    --pairs) PAIRS="$2"; shift 2;;
    --trims) TRIMS="$2"; shift 2;;
    --split_seeds) SPLIT_SEEDS="$2"; shift 2;;
    --model_seeds) MODEL_SEEDS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --enable_transformer) ENABLE_TRANSFORMER="$2"; shift 2;;
    --transformer_model) TRANSFORMER_MODEL="$2"; shift 2;;
    --transformer_offline) TRANSFORMER_OFFLINE="$2"; shift 2;;
    --transformer_cache_dir) TRANSFORMER_CACHE_DIR="$2"; shift 2;;
    --transformer_revision) TRANSFORMER_REVISION="$2"; shift 2;;
    --transformer_mode) TRANSFORMER_MODE="$2"; shift 2;;
    --transformer_device) TRANSFORMER_DEVICE="$2"; shift 2;;
    --transformer_freeze) TRANSFORMER_FREEZE="$2"; shift 2;;
    --transformer_unfreeze_last_n) TRANSFORMER_UNFREEZE_LAST_N="$2"; shift 2;;
    --transformer_fp16) TRANSFORMER_FP16="$2"; shift 2;;
    --transformer_batch_size) TRANSFORMER_BATCH_SIZE="$2"; shift 2;;
    --transformer_epochs) TRANSFORMER_EPOCHS="$2"; shift 2;;
    --transformer_patience) TRANSFORMER_PATIENCE="$2"; shift 2;;
    --transformer_lr) TRANSFORMER_LR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
 done

if [[ ! -d "$EXT_ROOT" ]]; then
  echo "Missing EXT_ROOT: $EXT_ROOT" >&2
  exit 1
fi

resolve_out_dir() {
  local pair="$1" trim="$2"
  local c1="$EXT_ROOT/$pair/trim$trim"
  local c2="$EXT_ROOT/$pair/out_plan8_trim$trim"
  if [[ -d "$c1" ]]; then echo "$c1"; return 0; fi
  if [[ -d "$c2" ]]; then echo "$c2"; return 0; fi
  return 1
}

for PAIR in $(echo "$PAIRS" | tr ',' ' '); do
  for TRIM in $(echo "$TRIMS" | tr ',' ' '); do
    if ! OUT_DIR="$(resolve_out_dir "$PAIR" "$TRIM")"; then
      echo "SKIP missing design dir: $EXT_ROOT/$PAIR/(trim$TRIM|out_plan8_trim$TRIM)" >&2
      continue
    fi
    echo "[RUN] strong models on $OUT_DIR" >&2
    "$SCRIPT_DIR/run_plan8_strong_models_parallel.sh" \
      --out_dir "$OUT_DIR" \
      --split_seeds "$SPLIT_SEEDS" \
      --model_seeds "$MODEL_SEEDS" \
      --jobs "$JOBS" \
      --enable_transformer "$ENABLE_TRANSFORMER" \
      --transformer_model "$TRANSFORMER_MODEL" \
      --transformer_offline "$TRANSFORMER_OFFLINE" \
      --transformer_cache_dir "$TRANSFORMER_CACHE_DIR" \
      --transformer_revision "$TRANSFORMER_REVISION" \
      --transformer_mode "$TRANSFORMER_MODE" \
      --transformer_device "$TRANSFORMER_DEVICE" \
      --transformer_freeze "$TRANSFORMER_FREEZE" \
      --transformer_unfreeze_last_n "$TRANSFORMER_UNFREEZE_LAST_N" \
      --transformer_fp16 "$TRANSFORMER_FP16" \
      --transformer_batch_size "$TRANSFORMER_BATCH_SIZE" \
      --transformer_epochs "$TRANSFORMER_EPOCHS" \
      --transformer_patience "$TRANSFORMER_PATIENCE" \
      --transformer_lr "$TRANSFORMER_LR"

    PYBIN="${PYBIN:-$(command -v python)}"
    $PYBIN "$SCRIPT_DIR/scripts/summarize_ci_plan8.py" \
      --out_dir "$OUT_DIR" \
      --out_json "$OUT_DIR/summary_vnext/ci_report_strong.json" \
      --n_boot 2000 \
      --seed 123
  done
 done

echo "DONE strong models for external designs." >&2
