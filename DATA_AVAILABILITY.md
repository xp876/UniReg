# Data Availability

This GitHub-ready repository includes the small raw input files needed to rerun
the core GSE83894 and GSE142696 retained-model analyses described in the v34
manuscript.

## Included raw inputs

```text
Data/raw/GSE83894/formatB_agg_only.zip
Data/raw/GSE83894/formatA_all_replicates.zip
Data/raw/GSE142696/GSM4237954_9MPRA_elements.fa.gz
Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.tsv.gz
Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz
```

File sizes and SHA256 checksums are recorded in:

```text
Data/raw_manifest.tsv
```

## Not included

The repository intentionally does not include full model checkpoints, complete
per-element prediction matrices, large intermediate work directories or the full
original analysis archives.

Optional motif/FIMO post hoc analyses require:

```text
Data/raw/motif/H14CORE_meme_format.meme
```

which is not included in this compact GitHub version.

## External comparator requirements

- `gkmsvm_optional` requires an external LS-GKM/gkm-SVM installation.
- `nt_transformer_delta_ens` requires transformer dependencies and access to the
  corresponding pretrained nucleotide-transformer model/cache.

The included `Results/` folder contains lightweight manuscript-ready summary
tables so readers can inspect the retained-model results without downloading the
full original project archive.

## Licensing note

The MIT License applies to the source code in this repository. Public raw data
files retain their original source terms and should be cited according to the
original GEO/public-data providers.
