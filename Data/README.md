# Data Folder

This folder contains the compact raw input files needed for the core manuscript
reruns.

```text
Data/
  raw/
    GSE83894/
      formatB_agg_only.zip
      formatA_all_replicates.zip
    GSE142696/
      GSM4237954_9MPRA_elements.fa.gz
      GSE142696_9MPRA.ActivityRatios.tsv.gz
      GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz
  raw_manifest.tsv
```

The benchmark code builds train/validation/test TSV files from these inputs.
The derived paired-lentiMPRA endpoint is:

```text
Delta = log2_WT - log2_MT
```

where WT is the integrated reporter-state measurement and MT is the episomal
reporter-state measurement.

The optional motif/FIMO database is not included in this compact release:

```text
Data/raw/motif/H14CORE_meme_format.meme
```

See `raw_manifest.tsv` for file sizes and SHA256 checksums.
