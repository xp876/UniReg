export MPLBACKEND=Agg
unset DISPLAY

# NatComm+ posthoc analyses (ceiling gap, residual structure, AND motif-level conv1 filter enrichment)

bash run_plan8_posthoc_analyses.sh \
  --main_out ./out_plan8 \
  --ext_root ./out_gse142696_plan8 \
  --trim_to_list 171,185 \
  --post_out ./posthoc_natcomm_plus \
  > 8_posthoc.o 2> 8_posthoc.e


# ------------------------------
# Nat Commun upgrade add-ons
# 1) Unified multi-split summaries (incl. paired p-values + BH-FDR)
# 2) Similarity robustness: drop top 1% most similar test elements
# ------------------------------

python scripts/summarize_ci_plan8.py \
  --root ./out_plan8 \
  --out_dir ./posthoc_natcomm_plus/summary/out_plan8 \
  --n_boot 5000 \
  --seed 0 \
  > 8b_summary_out_plan8.o 2> 8b_summary_out_plan8.e

mkdir -p ./posthoc_natcomm_plus/robustness_similarity

for PAT in \
  "cnn_wt_mt_delta3head_ens.test_predictions.tsv" \
  "cnn3head_kmer_fused_ens.test_predictions.tsv" \
  "kmer_delta_ens.test_predictions.tsv"; do

  OUTBASE=./posthoc_natcomm_plus/robustness_similarity/${PAT%%.test_predictions.tsv}

  # Some runs may not have every ensemble (e.g., fused model not generated in main out_plan8)
  if ! find ./out_plan8 -name "$PAT" -print -quit | grep -q .; then
    echo "[WARN] No prediction files found for pattern: $PAT under ./out_plan8 ; skipping similarity robustness." \
      > "${OUTBASE}.o"
    : > "${OUTBASE}.e"
    continue
  fi

  python scripts/summarize_similarity_robustness.py \
    --root ./out_plan8 \
    --model_pattern "$PAT" \
    --out_dir "$OUTBASE" \
    --q 0.99 \
    --k 6 \
    --n_boot 2000 \
    --seed 0 \
    > "${OUTBASE}.o" \
    2> "${OUTBASE}.e" || {
      echo "[WARN] similarity robustness failed for $PAT ; see ${OUTBASE}.e" >> "${OUTBASE}.e"
      # do not fail whole 8.sh for a non-critical add-on
      continue
    }
done

echo "[DONE] 8.sh completed (posthoc + summary + similarity robustness)"
exit 0
