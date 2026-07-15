# UniReg Model Architecture

UniReg is the readable manuscript name for:

```text
cnn3head_kmer_fused_ens
```

It is a task-specific late-fusion hybrid model for paired-lentiMPRA
sequence-to-function prediction. The model predicts:

```text
Delta = log2(WT / MT) = log2_WT - log2_MT
```

where WT denotes integrated reporter-state activity and MT denotes episomal
reporter-state activity.

UniReg is not a single monolithic neural network. It combines:

1. a paired three-head CNN ensemble, `cnn_wt_mt_delta3head_ens`;
2. a 6-mer k-mer ridge ensemble, `kmer_delta_ens`;
3. a validation-set constrained linear fusion of the two predictions.

## CNN Branch

The CNN branch is implemented in:

```text
Code/scripts/train_cnn_wt_mt_delta3head.py
Code/scripts/models.py
```

Architecture:

```text
Input tokens: B x L
Token mapping: A=0, C=1, G=2, T=3, N=4
Embedding: 5 x 32

Conv1d: 32  -> 128, kernel 7,  padding 3
Conv1d: 128 -> 128, kernel 7,  padding 3
Conv1d: 128 -> 128, kernel 13, padding 6

Activation: GELU
Pooling: global max pooling from all three convolution outputs
Feature concat: 128 * 3 = 384
Trunk: Linear(384 -> 128) + GELU + Dropout(0.25)

Heads:
  WT head:    Linear(128 -> 1)
  MT head:    Linear(128 -> 1)
  Delta head: Linear(128 -> 1)
```

The CNN branch jointly predicts WT, MT and Delta. This paired multi-head design
lets the model learn reporter-state-specific activity and the integration-
dependent contrast in the same shared sequence encoder.

Training defaults:

```text
epochs = 180
patience = 22
batch_size = 128
optimizer = AdamW
learning_rate = 3e-4
weight_decay = 1e-2
dropout = 0.25
loss = SmoothL1Loss / Huber, beta = 0.5
reverse-complement augmentation = enabled
EMA = enabled, decay = 0.999
Delta consistency lambda = 0.2
early stopping = validation Delta Pearson
```

The consistency term is:

```text
(pred_Delta - (pred_WT - pred_MT))^2
```

This term encourages the direct Delta head to remain consistent with the
difference between the two reporter-state heads.

## k-mer Ridge Branch

The k-mer branch is implemented in:

```text
Code/scripts/baseline_kmer_ridge.py
```

It uses:

```text
TfidfVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
Ridge regression
alpha candidates = 0.1, 1, 10, 100
selection metric = validation Pearson
```

This branch captures short-word sequence-composition signal, including
motif-like fragments and composition-accessible regulatory information that may
be highly predictive in some assay contexts.

## Seed-Level Ensembling

Seed-level ensembling is implemented in:

```text
Code/scripts/ensemble_predictions.py
```

The manuscript-level runs use:

```text
split_seeds = 0,1,2,3,4
model_seeds = 0,1,2,3,4
```

Within each split, model-seed predictions are aligned by `element_id` and
averaged. This reduces instability from random initialization and relatively
small MPRA benchmark datasets.

## CNN + k-mer Fusion

Fusion is implemented in:

```text
Code/scripts/fuse_two_models_linear.py
```

For validation labels `y`, CNN predictions `p1` and k-mer predictions `p2`, the
fusion script fits:

```text
d = p1 - p2
w = sum((y - p2) * d) / sum(d * d)
w = clamp(w, 0, 1)
pred = w * p1 + (1 - w) * p2
```

For the primary packaged GSE83894 split_seed0 result, the stored fusion weight
for model 1 is approximately:

```text
w = 0.9109362952167372
```

This value is split/task-specific and should not be treated as a universal fixed
weight.

## Conceptual Interpretation

UniReg integrates two complementary sources of sequence evidence:

- the CNN branch learns paired-assay-aware, position-sensitive regulatory
  grammar, including motif presence, local context, spacing and nonlinear
  patterns;
- the k-mer branch learns robust short-word composition signal.

The final validation-weighted fusion is intended to adapt across assay contexts:
when position-aware grammar is informative, the CNN can dominate; when
composition-accessible signal is stronger or more stable, the k-mer branch can
contribute more.
