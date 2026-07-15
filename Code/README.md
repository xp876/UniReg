# Code

This folder contains the core scripts needed to rerun the manuscript-level
retained-model UniReg benchmark.

Important entry points:

```text
run_plan8_all_single_node.sh      Primary GSE83894 benchmark wrapper
run_plan8_all_parallel.sh         Main primary benchmark runner
2.sh                              GSE142696 external benchmark launcher
run_plan8_gse142696_parallel.sh   GSE142696 design/trim runner
```

The Python scripts live in `Code/scripts/`. The most important model files are:

```text
scripts/train_cnn_wt_mt_delta3head.py
scripts/baseline_kmer_ridge.py
scripts/ensemble_predictions.py
scripts/fuse_two_models_linear.py
scripts/models.py
```

UniReg corresponds to `cnn3head_kmer_fused_ens`, which is produced by fusing:

```text
cnn_wt_mt_delta3head_ens
kmer_delta_ens
```

using validation-set constrained linear fusion.

