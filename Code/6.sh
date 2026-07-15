#!/usr/bin/env bash
set -euo pipefail

# Build cross-dataset / cross-design transfer matrix (default: main + external trim185).
# You may override ROOTS (comma-separated) or TRANSFER_TRIMS (e.g. 171,185).

export MPLBACKEND=${MPLBACKEND:-Agg}
unset DISPLAY || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYBIN=${PYBIN:-$(command -v python)}

OUTDIR=${OUTDIR:-./out_plan8/transfer}
TRANSFER_TRIMS=${TRANSFER_TRIMS:-185}
N_JOBS=${N_JOBS:-4}
SPLIT_TAG=${SPLIT_TAG:-split_seed0}
TRANSFER_SEED=${TRANSFER_SEED:-0}

mkdir -p "$OUTDIR"

ROOTS_CSV=${ROOTS:-}
if [[ -z "$ROOTS_CSV" ]]; then
  roots=()

  # Main benchmark (GSE83894)
  if [[ -d ./out_plan8/prepared/${SPLIT_TAG}/weighted ]]; then
    roots+=("./out_plan8")
  fi

  # External designs (GSE142696), default trim185; configurable via TRANSFER_TRIMS
  for pair in 3p3p 5p3p 5p5p; do
    for trim in $(echo "$TRANSFER_TRIMS" | tr ',' ' '); do
      cand="./out_gse142696_plan8/${pair}/out_plan8_trim${trim}"
      if [[ -d "$cand/prepared/${SPLIT_TAG}/weighted" ]]; then
        roots+=("$cand")
      fi
    done
  done

  if [[ ${#roots[@]} -eq 0 ]]; then
    echo "[ERROR] No valid transfer roots found. Expected prepared dirs under out_plan8 and/or out_gse142696_plan8/*/out_plan8_trim*/prepared/${SPLIT_TAG}/weighted" >&2
    exit 1
  fi

  ROOTS_CSV=$(IFS=,; echo "${roots[*]}")
fi

echo "[transfer6] ROOTS=$ROOTS_CSV" >&2
echo "[transfer6] OUTDIR=$OUTDIR" >&2

bash "$SCRIPT_DIR/run_plan8_transfer_matrix.sh"   --roots "$ROOTS_CSV"   --out_dir "$OUTDIR"   --split_tag "$SPLIT_TAG"   --seed "$TRANSFER_SEED"   --jobs "$N_JOBS"

# Legacy convenience symlink
if [[ ! -e ./transfer_plan8 ]]; then
  ln -s "$OUTDIR" ./transfer_plan8
fi

echo "Transfer matrix written to: $OUTDIR"
