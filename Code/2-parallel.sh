#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher: run all three GSE142696 construct designs in parallel on ONE node.
# Recommended usage:
#   JOBS_PER_DESIGN=4 bash 2-parallel.sh
# This will run:
#   3p3p, 5p3p, 5p5p
# in parallel, each with its own internal job pool (JOBS_PER_DESIGN).
#
# Logs are written to: ${OUT_ROOT:-./out_gse142696_plan8}/logs/

export MPLBACKEND=${MPLBACKEND:-Agg}
unset DISPLAY || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_ROOT=${OUT_ROOT:-./out_gse142696_plan8}
mkdir -p "$OUT_ROOT/logs"

( bash "$SCRIPT_DIR/2-1.sh" > "$OUT_ROOT/logs/2-1_3p3p.log" 2>&1 ) &
( bash "$SCRIPT_DIR/2-2.sh" > "$OUT_ROOT/logs/2-2_5p3p.log" 2>&1 ) &
( bash "$SCRIPT_DIR/2-3.sh" > "$OUT_ROOT/logs/2-3_5p5p.log" 2>&1 ) &

wait

echo "All designs finished. Logs in $OUT_ROOT/logs/"
