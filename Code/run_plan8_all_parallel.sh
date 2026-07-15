#!/usr/bin/env bash
set -euo pipefail

# Plan8 (GB/NC-ready): parallel runner for GSE83894 paired WT/MT -> delta benchmark
# - multi split_seeds + model_seeds
# - noise-aware weights (from FormatA)
# - baselines: k-mer ridge + (optional) LS-GKM
# - models: CNN delta + CNN WT/MT/delta joint (main)
# - per-prefix stratified eval + bootstrap CI + paired bootstrap
# - interpretability hooks

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


FORMATB_ZIP=""
FORMATA_ZIP=""
OUT_DIR=""
SPLIT_SEEDS="0,1,2,3,4"
MODEL_SEEDS="0,1,2,3,4"
AUTO_TRIM="yes"
TRIM_TO="0"
DELTA_CLIP_Q="0.01"
LOSS="huber"
RUN_GKMSVM="auto"    # auto|yes|no
N_JOBS="${N_JOBS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formatB_zip) FORMATB_ZIP="$2"; shift 2;;
    --formatA_zip) FORMATA_ZIP="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --split_seeds) SPLIT_SEEDS="$2"; shift 2;;
    --model_seeds) MODEL_SEEDS="$2"; shift 2;;
    --auto_trim) AUTO_TRIM="$2"; shift 2;;
    --trim_to) TRIM_TO="$2"; shift 2;;
    --delta_clip_q) DELTA_CLIP_Q="$2"; shift 2;;
    --loss) LOSS="$2"; shift 2;;
    --run_gkmsvm) RUN_GKMSVM="$2"; shift 2;;
    --jobs) N_JOBS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
 done

if [[ -z "$FORMATB_ZIP" || -z "$FORMATA_ZIP" || -z "$OUT_DIR" ]]; then
  echo "Missing required args. Need --formatB_zip --formatA_zip --out_dir";
  exit 1
fi

mkdir -p "$OUT_DIR"

# 0) ceilings + weights
$PYBIN "$SCRIPT_DIR/scripts/compute_replicate_ceiling_gb_by_prefix.py" \
  --data_zip "$FORMATA_ZIP" \
  --out_json "$OUT_DIR/ceiling_formatA_gb_by_prefix.json"

$PYBIN "$SCRIPT_DIR/scripts/compute_noise_weights_from_formatA_flexible.py" \
  --data_zip "$FORMATA_ZIP" \
  --out_tsv "$OUT_DIR/noise_weights_from_formatA.tsv"

IFS=',' read -ra SPLITS <<< "$SPLIT_SEEDS"
IFS=',' read -ra MSEEDS <<< "$MODEL_SEEDS"

# helper: run a list of commands in parallel (single-node safe)
# We intentionally avoid login shells. Jobs are executed via a Python pool using
# the same interpreter as this runner, ensuring the same conda env (torch, etc.).
run_parallel() {
  local jobfile="$1"
  local nj="$2"
  local log_dir="$3"
  mkdir -p "$log_dir"
  "$PYBIN" "$SCRIPT_DIR/scripts/run_jobfile_pool.py" --jobfile "$jobfile" --jobs "$nj" --log_dir "$log_dir"
}

for SPLIT_SEED in "${SPLITS[@]}"; do
  SPLIT_TAG="split_seed${SPLIT_SEED}"
  echo "==== [SPLIT] $SPLIT_TAG ===="

  $PYBIN "$SCRIPT_DIR/scripts/make_splits_elementwise.py" \
    --data_zip "$FORMATB_ZIP" \
    --out_json "$OUT_DIR/splits_${SPLIT_TAG}.json" \
    --seed "$SPLIT_SEED" \
    --require_paired

  PREP_RAW="$OUT_DIR/prepared/$SPLIT_TAG/raw"
  PREP_W="$OUT_DIR/prepared/$SPLIT_TAG/weighted"

  $PYBIN "$SCRIPT_DIR/scripts/prepare_paired_dataset.py" \
    --data_zip "$FORMATB_ZIP" \
    --splits_json "$OUT_DIR/splits_${SPLIT_TAG}.json" \
    --out_dir "$PREP_RAW" \
    --auto_trim "$AUTO_TRIM" \
    --trim_to "$TRIM_TO"

  $PYBIN "$SCRIPT_DIR/scripts/attach_noise_weights.py" \
    --prepared_dir "$PREP_RAW" \
    --weights_tsv "$OUT_DIR/noise_weights_from_formatA.tsv" \
    --out_dir "$PREP_W"

  $PYBIN "$SCRIPT_DIR/scripts/analyze_targets.py" --prepared_dir "$PREP_W" --out_dir "$OUT_DIR/analysis/$SPLIT_TAG" --split test

  JOBS_DIR="$OUT_DIR/jobs/$SPLIT_TAG"
  mkdir -p "$JOBS_DIR"
  JOBFILE="$JOBS_DIR/train_jobs.txt"
  : > "$JOBFILE"

  for SEED in "${MSEEDS[@]}"; do
    ROOT="$OUT_DIR/results/$SPLIT_TAG/seed$SEED"
    mkdir -p "$ROOT"

    # Baselines
    echo "$PYBIN '$SCRIPT_DIR/scripts/baseline_kmer_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_delta' --target delta --seed '$SEED' --k 6" >> "$JOBFILE"
    echo "$PYBIN '$SCRIPT_DIR/scripts/baseline_onehot_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/onehot_ridge_delta' --target delta --seed '$SEED'" >> "$JOBFILE"
    echo "$PYBIN '$SCRIPT_DIR/scripts/baseline_kmer_sgd_elasticnet.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_elasticnet_delta' --target delta --seed '$SEED' --k 6" >> "$JOBFILE"
    echo "$PYBIN '$SCRIPT_DIR/scripts/baseline_kmer_nystroem_ridge.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/kmer_nystroem_ridge_delta' --target delta --seed '$SEED' --k 6 --n_components 2000 --gamma 0.001 --alpha 1.0" >> "$JOBFILE"

    echo "$PYBIN '$SCRIPT_DIR/scripts/train_cnn_single.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_delta' --target delta --seed '$SEED' --rc_aug" >> "$JOBFILE"

    echo "$PYBIN '$SCRIPT_DIR/scripts/train_cnn_wt_mt_delta.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_wt_mt_delta' --seed '$SEED' --rc_aug --loss '$LOSS' --delta_clip_q '$DELTA_CLIP_Q' --stop_on delta" >> "$JOBFILE"

    # 3-head WT/MT/Δ CNN with Δ-consistency regularizer (often improves Δ stability)
    echo "$PYBIN '$SCRIPT_DIR/scripts/train_cnn_wt_mt_delta3head.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_wt_mt_delta3head' --seed '$SEED' --rc_aug --use_ema --loss '$LOSS' --stop_on delta --delta_consistency_lambda 0.2" >> "$JOBFILE"

    # "physically consistent" variant: predict WT/MT then derive Δ (no direct Δ head)
    echo "$PYBIN '$SCRIPT_DIR/scripts/train_cnn_wt_mt_derive_delta.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_wt_mt_derive_delta' --seed '$SEED' --rc_aug --use_ema --loss '$LOSS' --stop_on delta" >> "$JOBFILE"

    echo "$PYBIN '$SCRIPT_DIR/scripts/train_cnn_mean_delta.py' --prepared_dir '$PREP_W' --out_dir '$ROOT/cnn_mean_delta' --seed '$SEED' --rc_aug --stop_on delta" >> "$JOBFILE"
  done

  # gkm-SVM once per split (not seed dependent)
  if [[ "$RUN_GKMSVM" == "yes" || "$RUN_GKMSVM" == "auto" ]]; then
    echo "$PYBIN '$SCRIPT_DIR/scripts/baseline_gkmsvm_optional.py' --prepared_dir '$PREP_W' --out_dir '$OUT_DIR/results/$SPLIT_TAG/gkmsvm_optional'" >> "$JOBFILE"
  fi

  echo "-- [PARALLEL TRAIN] $SPLIT_TAG jobs=$(wc -l < "$JOBFILE")  N_JOBS=$N_JOBS"
  run_parallel "$JOBFILE" "$N_JOBS" "$JOBS_DIR/_logs"

  # stratified eval for key models (per seed)
  for SEED in "${MSEEDS[@]}"; do
    ROOT="$OUT_DIR/results/$SPLIT_TAG/seed$SEED"
    $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
      --pred_tsv "$ROOT/cnn_wt_mt_delta/cnn_wt_mt_delta.test_predictions.tsv" \
      --out_json "$ROOT/cnn_wt_mt_delta/cnn_wt_mt_delta.delta.stratified.vnext.json" \
      --y_col y_delta --pred_col pred_delta \
      --exclude_prefixes C

    if [[ -f "$ROOT/cnn_wt_mt_delta3head/cnn_wt_mt_delta3head.test_predictions.tsv" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
        --pred_tsv "$ROOT/cnn_wt_mt_delta3head/cnn_wt_mt_delta3head.test_predictions.tsv" \
        --out_json "$ROOT/cnn_wt_mt_delta3head/cnn_wt_mt_delta3head.delta.stratified.vnext.json" \
        --y_col y_delta --pred_col pred_delta \
        --exclude_prefixes C
    fi

    $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
      --pred_tsv "$ROOT/kmer_delta/kmer_delta.test_predictions.tsv" \
      --out_json "$ROOT/kmer_delta/kmer_delta.delta.stratified.vnext.json" \
      --y_col y_delta --pred_col pred_delta \
      --exclude_prefixes C

    if [[ -f "$ROOT/onehot_ridge_delta/onehot_ridge_delta.test_predictions.tsv" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
        --pred_tsv "$ROOT/onehot_ridge_delta/onehot_ridge_delta.test_predictions.tsv" \
        --out_json "$ROOT/onehot_ridge_delta/onehot_ridge_delta.delta.stratified.vnext.json" \
        --y_col y_delta --pred_col pred_delta \
        --exclude_prefixes C
    fi

    if [[ -f "$ROOT/kmer_elasticnet_delta/kmer_elasticnet_delta.test_predictions.tsv" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
        --pred_tsv "$ROOT/kmer_elasticnet_delta/kmer_elasticnet_delta.test_predictions.tsv" \
        --out_json "$ROOT/kmer_elasticnet_delta/kmer_elasticnet_delta.delta.stratified.vnext.json" \
        --y_col y_delta --pred_col pred_delta \
        --exclude_prefixes C
    fi

    if [[ -f "$ROOT/kmer_nystroem_ridge_delta/kmer_nystroem_ridge_delta.test_predictions.tsv" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
        --pred_tsv "$ROOT/kmer_nystroem_ridge_delta/kmer_nystroem_ridge_delta.test_predictions.tsv" \
        --out_json "$ROOT/kmer_nystroem_ridge_delta/kmer_nystroem_ridge_delta.delta.stratified.vnext.json" \
        --y_col y_delta --pred_col pred_delta \
        --exclude_prefixes C
    fi

    if [[ -f "$ROOT/cnn_wt_mt_derive_delta/cnn_wt_mt_derive_delta.test_predictions.tsv" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
        --pred_tsv "$ROOT/cnn_wt_mt_derive_delta/cnn_wt_mt_derive_delta.test_predictions.tsv" \
        --out_json "$ROOT/cnn_wt_mt_derive_delta/cnn_wt_mt_derive_delta.delta.stratified.vnext.json" \
        --y_col y_delta --pred_col pred_delta \
        --exclude_prefixes C
    fi
  done

  if [[ -f "$OUT_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.test_predictions.tsv" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/evaluate_stratified_vnext.py" \
      --pred_tsv "$OUT_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.test_predictions.tsv" \
      --out_json "$OUT_DIR/results/$SPLIT_TAG/gkmsvm_optional/gkmsvm_optional.delta.stratified.vnext.json" \
      --y_col y_delta --pred_col pred_delta \
      --exclude_prefixes C
  fi

  # ensemble across model_seeds for this split
  echo "-- [ENSEMBLE] $SPLIT_TAG"
  ENS_ROOT="$OUT_DIR/results/$SPLIT_TAG/ensemble"
  mkdir -p "$ENS_ROOT"

  make_list() {
    local pat="$1"
    local list=""
    for SEED in "${MSEEDS[@]}"; do
      local p="$OUT_DIR/results/$SPLIT_TAG/seed$SEED/$pat"
      if [[ -f "$p" ]]; then
        list+="$p,"
      fi
    done
    echo "${list%,}"
  }

  # kmer ensemble
  KMER_LIST=$(make_list "kmer_delta/kmer_delta.test_predictions.tsv")
  if [[ -n "$KMER_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$KMER_LIST" \
      --out_dir "$ENS_ROOT/kmer_delta_ens" \
      --model_name "kmer_delta_ens" \
      --y_cols "y_delta" \
      --pred_cols "pred_delta"

    KMER_VAL_LIST=$(make_list "kmer_delta/kmer_delta.val_predictions.tsv")
    if [[ -n "$KMER_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$KMER_VAL_LIST" \
        --out_dir "$ENS_ROOT/kmer_delta_ens" \
        --model_name "kmer_delta_ens_val" \
        --out_pred_tsv "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.val_predictions.tsv" \
        --y_cols "y_delta" \
        --pred_cols "pred_delta"
    fi
  fi

  # onehot ridge ensemble
  ONEHOT_LIST=$(make_list "onehot_ridge_delta/onehot_ridge_delta.test_predictions.tsv")
  if [[ -n "$ONEHOT_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$ONEHOT_LIST" \
      --out_dir "$ENS_ROOT/onehot_ridge_delta_ens" \
      --model_name "onehot_ridge_delta_ens" \
      --y_cols "y_delta" \
      --pred_cols "pred_delta"

    ONEHOT_VAL_LIST=$(make_list "onehot_ridge_delta/onehot_ridge_delta.val_predictions.tsv")
    if [[ -n "$ONEHOT_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$ONEHOT_VAL_LIST" \
        --out_dir "$ENS_ROOT/onehot_ridge_delta_ens" \
        --model_name "onehot_ridge_delta_ens_val" \
        --out_pred_tsv "$ENS_ROOT/onehot_ridge_delta_ens/onehot_ridge_delta_ens.val_predictions.tsv" \
        --y_cols "y_delta" \
        --pred_cols "pred_delta"
    fi
  fi

  # k-mer elasticnet (SGD) ensemble
  KMEREN_LIST=$(make_list "kmer_elasticnet_delta/kmer_elasticnet_delta.test_predictions.tsv")
  if [[ -n "$KMEREN_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$KMEREN_LIST" \
      --out_dir "$ENS_ROOT/kmer_elasticnet_delta_ens" \
      --model_name "kmer_elasticnet_delta_ens" \
      --y_cols "y_delta" \
      --pred_cols "pred_delta"

    KMEREN_VAL_LIST=$(make_list "kmer_elasticnet_delta/kmer_elasticnet_delta.val_predictions.tsv")
    if [[ -n "$KMEREN_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$KMEREN_VAL_LIST" \
        --out_dir "$ENS_ROOT/kmer_elasticnet_delta_ens" \
        --model_name "kmer_elasticnet_delta_ens_val" \
        --out_pred_tsv "$ENS_ROOT/kmer_elasticnet_delta_ens/kmer_elasticnet_delta_ens.val_predictions.tsv" \
        --y_cols "y_delta" \
        --pred_cols "pred_delta"
    fi
  fi

  # k-mer Nystroem-RBF ridge ensemble
  KMERNY_LIST=$(make_list "kmer_nystroem_ridge_delta/kmer_nystroem_ridge_delta.test_predictions.tsv")
  if [[ -n "$KMERNY_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$KMERNY_LIST" \
      --out_dir "$ENS_ROOT/kmer_nystroem_ridge_delta_ens" \
      --model_name "kmer_nystroem_ridge_delta_ens" \
      --y_cols "y_delta" \
      --pred_cols "pred_delta"

    KMERNY_VAL_LIST=$(make_list "kmer_nystroem_ridge_delta/kmer_nystroem_ridge_delta.val_predictions.tsv")
    if [[ -n "$KMERNY_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$KMERNY_VAL_LIST" \
        --out_dir "$ENS_ROOT/kmer_nystroem_ridge_delta_ens" \
        --model_name "kmer_nystroem_ridge_delta_ens_val" \
        --out_pred_tsv "$ENS_ROOT/kmer_nystroem_ridge_delta_ens/kmer_nystroem_ridge_delta_ens.val_predictions.tsv" \
        --y_cols "y_delta" \
        --pred_cols "pred_delta"
    fi
  fi

  CNN_DELTA_LIST=$(make_list "cnn_delta/cnn_delta.test_predictions.tsv")
  if [[ -n "$CNN_DELTA_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$CNN_DELTA_LIST" \
      --out_dir "$ENS_ROOT/cnn_delta_ens" \
      --model_name "cnn_delta_ens" \
      --y_cols "y_delta" \
      --pred_cols "pred_delta"
  fi

  JOINT_LIST=$(make_list "cnn_wt_mt_delta/cnn_wt_mt_delta.test_predictions.tsv")
  if [[ -n "$JOINT_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$JOINT_LIST" \
      --out_dir "$ENS_ROOT/cnn_wt_mt_delta_ens" \
      --model_name "cnn_wt_mt_delta_ens" \
      --y_cols "y_delta,y_int,y_epi" \
      --pred_cols "pred_delta,pred_int,pred_epi"

    JOINT_VAL_LIST=$(make_list "cnn_wt_mt_delta/cnn_wt_mt_delta.val_predictions.tsv")
    if [[ -n "$JOINT_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$JOINT_VAL_LIST" \
        --out_dir "$ENS_ROOT/cnn_wt_mt_delta_ens" \
        --model_name "cnn_wt_mt_delta_ens_val" \
        --out_pred_tsv "$ENS_ROOT/cnn_wt_mt_delta_ens/cnn_wt_mt_delta_ens.val_predictions.tsv" \
        --y_cols "y_delta,y_int,y_epi" \
        --pred_cols "pred_delta,pred_int,pred_epi"
    fi
  fi

  JOINT3_LIST=$(make_list "cnn_wt_mt_delta3head/cnn_wt_mt_delta3head.test_predictions.tsv")
  if [[ -n "$JOINT3_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$JOINT3_LIST" \
      --out_dir "$ENS_ROOT/cnn_wt_mt_delta3head_ens" \
      --model_name "cnn_wt_mt_delta3head_ens" \
      --y_cols "y_delta,y_int,y_epi" \
      --pred_cols "pred_delta,pred_int,pred_epi"

    JOINT3_VAL_LIST=$(make_list "cnn_wt_mt_delta3head/cnn_wt_mt_delta3head.val_predictions.tsv")
    if [[ -n "$JOINT3_VAL_LIST" ]]; then
      $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
        --pred_tsvs "$JOINT3_VAL_LIST" \
        --out_dir "$ENS_ROOT/cnn_wt_mt_delta3head_ens" \
        --model_name "cnn_wt_mt_delta3head_ens_val" \
        --out_pred_tsv "$ENS_ROOT/cnn_wt_mt_delta3head_ens/cnn_wt_mt_delta3head_ens.val_predictions.tsv" \
        --y_cols "y_delta,y_int,y_epi" \
        --pred_cols "pred_delta,pred_int,pred_epi"
    fi
  fi

  # derive-Δ ensemble
  DERIVE_LIST=$(make_list "cnn_wt_mt_derive_delta/cnn_wt_mt_derive_delta.test_predictions.tsv")
  if [[ -n "$DERIVE_LIST" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/ensemble_predictions.py" \
      --pred_tsvs "$DERIVE_LIST" \
      --out_dir "$ENS_ROOT/cnn_wt_mt_derive_delta_ens" \
      --model_name "cnn_wt_mt_derive_delta_ens" \
      --y_cols "y_delta,y_int,y_epi" \
      --pred_cols "pred_delta,pred_int,pred_epi"
  fi

  # CNN + k-mer fusion (weight fit on val ensemble, applied to test ensemble)
  if [[ -f "$ENS_ROOT/cnn_wt_mt_delta_ens/cnn_wt_mt_delta_ens.val_predictions.tsv" && -f "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.val_predictions.tsv" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/fuse_two_models_linear.py" \
      --val_pred1 "$ENS_ROOT/cnn_wt_mt_delta_ens/cnn_wt_mt_delta_ens.val_predictions.tsv" \
      --val_pred2 "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.val_predictions.tsv" \
      --test_pred1 "$ENS_ROOT/cnn_wt_mt_delta_ens/cnn_wt_mt_delta_ens.test_predictions.tsv" \
      --test_pred2 "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.test_predictions.tsv" \
      --y_col y_delta --p_col1 pred_delta --p_col2 pred_delta \
      --constrain_01 \
      --out_dir "$ENS_ROOT/cnn_kmer_fused_ens" \
      --name "cnn_kmer_fused_ens" \
      --out_pred_col pred_delta
  fi

  if [[ -f "$ENS_ROOT/cnn_wt_mt_delta3head_ens/cnn_wt_mt_delta3head_ens.val_predictions.tsv" && -f "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.val_predictions.tsv" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/fuse_two_models_linear.py" \
      --val_pred1 "$ENS_ROOT/cnn_wt_mt_delta3head_ens/cnn_wt_mt_delta3head_ens.val_predictions.tsv" \
      --val_pred2 "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.val_predictions.tsv" \
      --test_pred1 "$ENS_ROOT/cnn_wt_mt_delta3head_ens/cnn_wt_mt_delta3head_ens.test_predictions.tsv" \
      --test_pred2 "$ENS_ROOT/kmer_delta_ens/kmer_delta_ens.test_predictions.tsv" \
      --y_col y_delta --p_col1 pred_delta --p_col2 pred_delta \
      --constrain_01 \
      --out_dir "$ENS_ROOT/cnn3head_kmer_fused_ens" \
      --name "cnn3head_kmer_fused_ens" \
      --out_pred_col pred_delta
  fi

done

# summarize (mean/std across split seeds)
$PYBIN "$SCRIPT_DIR/scripts/summarize_gb_vnext.py" --root "$OUT_DIR" --out_dir "$OUT_DIR/summary_vnext"

# bootstrap CI + paired bootstrap (paper-critical)
mkdir -p "$OUT_DIR/summary_plan8"
$PYBIN "$SCRIPT_DIR/scripts/summarize_ci_plan8.py" --root "$OUT_DIR" --out_dir "$OUT_DIR/summary_plan8" --n_boot 2000 --seed 0

# interpretability demo on reference run
REF_SPLIT_TAG="split_seed0"
REF_SEED="0"
REF_CKPT="$OUT_DIR/results/$REF_SPLIT_TAG/seed$REF_SEED/cnn_wt_mt_delta/cnn_wt_mt_delta_best.pt"

if [[ -f "$REF_CKPT" ]]; then
  echo "==== [INTERPRET] using $REF_CKPT ===="
  $PYBIN "$SCRIPT_DIR/scripts/explain_conv1_pwm.py" --ckpt "$REF_CKPT" --out_dir "$OUT_DIR/explain_plan8" --max_filters 32
  $PYBIN "$SCRIPT_DIR/scripts/export_pwm_to_meme.py" --pwm_tsv "$OUT_DIR/explain_plan8/conv1_effective_pwm.tsv" --out_meme "$OUT_DIR/explain_plan8/conv1_filters.meme"

  PREP_REF="$OUT_DIR/prepared/$REF_SPLIT_TAG/weighted"
  # R-only case studies (top +/- delta)
  $PYBIN "$SCRIPT_DIR/scripts/ism_delta_gb.py" \
    --prepared_dir "$PREP_REF" \
    --ckpt "$REF_CKPT" \
    --out_dir "$OUT_DIR/ism_R_cases" \
    --split test \
    --prefix R \
    --top_pos 6 --top_neg 6

  # controls localization sanity check
  $PYBIN "$SCRIPT_DIR/scripts/ism_delta_gb.py" \
    --prepared_dir "$PREP_REF" \
    --ckpt "$REF_CKPT" \
    --out_dir "$OUT_DIR/ism_controls" \
    --split test \
    --only_controls \
    --top_pos 6 --top_neg 6

  if [[ -f "$OUT_DIR/ism_controls/ism_summary_all.tsv" ]]; then
    $PYBIN "$SCRIPT_DIR/scripts/evaluate_ism_controls.py" \
      --ism_summary "$OUT_DIR/ism_controls/ism_summary_all.tsv" \
      --out_tsv "$OUT_DIR/ism_controls/ism_controls.localization.tsv" \
      --topk 10
  fi
else
  echo "WARN: reference checkpoint not found; skip interpretability. Expected $REF_CKPT"
fi

echo "DONE Plan8. Outputs in: $OUT_DIR"
