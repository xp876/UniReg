#!/usr/bin/env bash
set -euo pipefail

# Incremental runner for Klein 2020 / GSE142696 extra episomal assays.
# It does NOT rerun your completed GSE83894 or 3p3p/5p3p/5p5p pipeline.
# It creates a new parallel side-car result tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYBIN="${PYBIN:-$(command -v python)}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
unset DISPLAY || true

# Existing project scripts from unireg2
CORE_SCRIPT_DIR="${CORE_SCRIPT_DIR:-$PROJECT_ROOT}"
if [[ ! -d "$CORE_SCRIPT_DIR/scripts" ]]; then
  echo "[ERROR] Could not find existing project scripts/ under CORE_SCRIPT_DIR=$CORE_SCRIPT_DIR" >&2
  echo "Set CORE_SCRIPT_DIR to the root of your extracted unireg2 project." >&2
  exit 1
fi
export PYTHONPATH="$CORE_SCRIPT_DIR/scripts:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/external_data/gse142696_klein2020}"
ELEMENTS_FA="${ELEMENTS_FA:-$DATA_DIR/GSM4237954_9MPRA_elements.fa.gz}"
MEAN_TSV="${MEAN_TSV:-$DATA_DIR/GSE142696_9MPRA.ActivityRatios.tsv.gz}"
REPS_TSV="${REPS_TSV:-$DATA_DIR/GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/out_gse142696_episomal_panel}"

ASSAYS="${ASSAYS:-pGL4,HSS,ORI}"
TRIM_TO_LIST="${TRIM_TO_LIST:-171,185}"
SPLIT_SEEDS="${SPLIT_SEEDS:-0,1,2,3,4}"
MODEL_SEEDS="${MODEL_SEEDS:-0,1,2,3,4}"
MODELS="${MODELS:-kmer_ridge,onehot_ridge,kmer_elasticnet,cnn_single}"
N_JOBS="${N_JOBS:-4}"
RUN_GKMSVM="${RUN_GKMSVM:-no}"

mkdir -p "$OUT_ROOT"
IFS=',' read -ra ASSAY_ARR <<< "$ASSAYS"
IFS=',' read -ra TRIM_ARR <<< "$TRIM_TO_LIST"
IFS=',' read -ra SPLIT_ARR <<< "$SPLIT_SEEDS"
IFS=',' read -ra MSEED_ARR <<< "$MODEL_SEEDS"
IFS=',' read -ra MODEL_ARR <<< "$MODELS"

run_parallel() {
  local jobfile="$1"
  local nj="$2"
  local log_dir="$3"
  mkdir -p "$log_dir"
  "$PYBIN" "$CORE_SCRIPT_DIR/scripts/run_jobfile_pool.py" --jobfile "$jobfile" --jobs "$nj" --log_dir "$log_dir"
}

for ASSAY in "${ASSAY_ARR[@]}"; do
  echo "==== [ASSAY] $ASSAY ===="
  ASSAY_DIR="$OUT_ROOT/$ASSAY"
  mkdir -p "$ASSAY_DIR/exports"

  "$PYBIN" "$SCRIPT_DIR/gse142696_build_single_assay_exports.py" \
    --assay "$ASSAY" \
    --elements_fa "$ELEMENTS_FA" \
    --activity_mean_tsv "$MEAN_TSV" \
    --activity_reps_tsv "$REPS_TSV" \
    --trim_len 185 \
    --out_dir "$ASSAY_DIR/exports"

  FORMATB="$ASSAY_DIR/exports/formatB_agg_only.zip"
  FORMATA="$ASSAY_DIR/exports/formatA_all_replicates.zip"

  for TRIM in "${TRIM_ARR[@]}"; do
    echo "---- [RUN] $ASSAY / trim${TRIM} ----"
    RUN_DIR="$ASSAY_DIR/out_plan8_trim${TRIM}"
    mkdir -p "$RUN_DIR"

    "$PYBIN" "$SCRIPT_DIR/compute_replicate_ceiling_single_by_prefix.py" \
      --data_zip "$FORMATA" \
      --out_json "$RUN_DIR/ceiling_formatA_single.json"

    "$PYBIN" "$SCRIPT_DIR/compute_noise_weights_single_from_formatA.py" \
      --data_zip "$FORMATA" \
      --out_tsv "$RUN_DIR/noise_weights_from_formatA.tsv"

    for SPLIT_SEED in "${SPLIT_ARR[@]}"; do
      SPLIT_TAG="split_seed${SPLIT_SEED}"
      "$PYBIN" "$CORE_SCRIPT_DIR/scripts/make_splits_elementwise.py" \
        --data_zip "$FORMATB" \
        --out_json "$RUN_DIR/splits_${SPLIT_TAG}.json" \
        --seed "$SPLIT_SEED"

      PREP_RAW="$RUN_DIR/prepared/$SPLIT_TAG/raw"
      PREP_W="$RUN_DIR/prepared/$SPLIT_TAG/weighted"
      "$PYBIN" "$SCRIPT_DIR/prepare_single_assay_dataset.py" \
        --data_zip "$FORMATB" \
        --splits_json "$RUN_DIR/splits_${SPLIT_TAG}.json" \
        --out_dir "$PREP_RAW" \
        --auto_trim yes \
        --trim_to "$TRIM"

      "$PYBIN" "$CORE_SCRIPT_DIR/scripts/attach_noise_weights.py" \
        --prepared_dir "$PREP_RAW" \
        --weights_tsv "$RUN_DIR/noise_weights_from_formatA.tsv" \
        --out_dir "$PREP_W"

      JOBS_DIR="$RUN_DIR/jobs/$SPLIT_TAG"
      mkdir -p "$JOBS_DIR"
      JOBFILE="$JOBS_DIR/train_jobs.txt"
      : > "$JOBFILE"

      for SEED in "${MSEED_ARR[@]}"; do
        ROOT="$RUN_DIR/results/$SPLIT_TAG/seed$SEED"
        mkdir -p "$ROOT"
        for MODEL in "${MODEL_ARR[@]}"; do
          case "$MODEL" in
            kmer_ridge)
              echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/baseline_kmer_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_mean' --target mean --seed '$SEED' --k 6" >> "$JOBFILE"
              ;;
            onehot_ridge)
              echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/baseline_onehot_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/onehot_ridge_mean' --target mean --seed '$SEED'" >> "$JOBFILE"
              ;;
            kmer_elasticnet)
              echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/baseline_kmer_sgd_elasticnet.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_elasticnet_mean' --target mean --seed '$SEED' --k 6" >> "$JOBFILE"
              ;;
            kmer_nystroem_ridge)
              echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/baseline_kmer_nystroem_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_nystroem_ridge_mean' --target mean --seed '$SEED' --k 6 --n_components 2000 --gamma 0.001 --alpha 1.0" >> "$JOBFILE"
              ;;
            cnn_single)
              echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/train_cnn_single.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_mean' --target mean --seed '$SEED' --rc_aug" >> "$JOBFILE"
              ;;
            gkmsvm_optional)
              :
              ;;
            *)
              echo "[ERROR] Unknown model '$MODEL'" >&2; exit 1 ;;
          esac
        done
      done

      if [[ "$RUN_GKMSVM" == "yes" || "$RUN_GKMSVM" == "auto" ]]; then
        echo "$PYBIN '$CORE_SCRIPT_DIR/scripts/baseline_gkmsvm_optional.py' --prepared_dir '$PREP_W' --out_dir '$RUN_DIR/results/$SPLIT_TAG/gkmsvm_optional' --target mean" >> "$JOBFILE"
      fi

      echo "-- [PARALLEL TRAIN] $ASSAY trim${TRIM} $SPLIT_TAG jobs=$(wc -l < "$JOBFILE")"
      run_parallel "$JOBFILE" "$N_JOBS" "$JOBS_DIR/_logs"

      ENS_ROOT="$RUN_DIR/results/$SPLIT_TAG/ensemble"
      mkdir -p "$ENS_ROOT"
      make_list() {
        local pat="$1"
        local list=""
        for SEED in "${MSEED_ARR[@]}"; do
          local p="$RUN_DIR/results/$SPLIT_TAG/seed$SEED/$pat"
          if [[ -f "$p" ]]; then list+="$p,"; fi
        done
        echo "${list%,}"
      }

      declare -A model_to_pat=(
        [kmer_mean_ens]="kmer_mean/kmer_mean.test_predictions.tsv"
        [onehot_ridge_mean_ens]="onehot_ridge_mean/onehot_ridge_mean.test_predictions.tsv"
        [kmer_elasticnet_mean_ens]="kmer_elasticnet_mean/kmer_elasticnet_mean.test_predictions.tsv"
        [kmer_nystroem_ridge_mean_ens]="kmer_nystroem_ridge_mean/kmer_nystroem_ridge_mean.test_predictions.tsv"
        [cnn_mean_ens]="cnn_mean/cnn_mean.test_predictions.tsv"
      )
      for ENS_NAME in "${!model_to_pat[@]}"; do
        LIST=$(make_list "${model_to_pat[$ENS_NAME]}")
        if [[ -n "$LIST" && "$LIST" == *,* ]]; then
          "$PYBIN" "$CORE_SCRIPT_DIR/scripts/ensemble_predictions.py" \
            --pred_tsvs "$LIST" \
            --out_dir "$ENS_ROOT/$ENS_NAME" \
            --model_name "$ENS_NAME" \
            --y_cols y_mean \
            --pred_cols pred_mean
        fi
      done

      # stratified eval on ensembles
      for ENS_NAME in cnn_mean_ens kmer_mean_ens onehot_ridge_mean_ens kmer_elasticnet_mean_ens kmer_nystroem_ridge_mean_ens; do
        PRED="$ENS_ROOT/$ENS_NAME/${ENS_NAME}.test_predictions.tsv"
        [[ -f "$PRED" ]] || continue
        "$PYBIN" "$CORE_SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
          --pred_tsv "$PRED" \
          --out_json "$ENS_ROOT/$ENS_NAME/${ENS_NAME}.mean.stratified.vnext.json" \
          --y_col y_mean --pred_col pred_mean
      done
      if [[ -f "$RUN_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.test_predictions.tsv" ]]; then
        "$PYBIN" "$CORE_SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
          --pred_tsv "$RUN_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.test_predictions.tsv" \
          --out_json "$RUN_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.mean.stratified.vnext.json" \
          --y_col y_mean --pred_col pred_mean
      fi
    done
  done
done

SUMMARY_DIR="$OUT_ROOT/summary"
mkdir -p "$SUMMARY_DIR"
"$PYBIN" "$SCRIPT_DIR/summarize_single_assay_panel.py" --root "$OUT_ROOT" --out_dir "$SUMMARY_DIR"

echo "[ok] Completed incremental episomal-panel run. Root: $OUT_ROOT"
