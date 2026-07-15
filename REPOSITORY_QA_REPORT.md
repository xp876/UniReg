# Repository QA Report

Generated for the GitHub-ready UniReg directory on 2026-07-15.

## Summary

This directory is intended to be uploaded as the public UniReg GitHub repository
for manuscript-level reproducibility of the v34 retained-model analyses.

## Checks performed

- Required root files are present:
  - `README.md`
  - `LICENSE`
  - `DATA_AVAILABILITY.md`
  - `GITHUB_UPLOAD_CHECKLIST.md`
  - `environment.yml`
  - `requirements.txt`
- Required raw input files are present:
  - `Data/raw/GSE83894/formatB_agg_only.zip`
  - `Data/raw/GSE83894/formatA_all_replicates.zip`
  - `Data/raw/GSE142696/GSM4237954_9MPRA_elements.fa.gz`
  - `Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.tsv.gz`
  - `Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz`
- Checksums were generated in `Data/raw_manifest.tsv`.
- The latest user-provided v3 schematic was installed as `Docs/Unireg.pdf`,
  preserved as `Docs/unireg-v3.pdf`, and rendered to `Docs/Unireg.png` for
  GitHub README display.
- Duplicate/copy schematic files were removed from the clean release directory.
- Python bytecode cache directories were removed from the clean release
  directory.
- Python files under `Code/scripts/` were byte-compile checked successfully.
- The required raw data files were checked against `.gitignore` and are
  trackable.

## Directory size

The final directory contains the compact code, raw inputs, lightweight summary
results, documentation and supplementary ABD materials needed for
manuscript-level reproducibility. See the current file count and size from the
final QA summary if this directory is regenerated.

## Known intentional exclusions

- Full model checkpoints are not included.
- Complete per-element prediction matrices are not included.
- Large intermediate work directories are not included.
- The optional FIMO motif database `H14CORE_meme_format.meme` is not included.
- The full ABD execution bundle is not included; lightweight ABD code and
  summary results are included.

## Notes

The repository is designed for compact manuscript-level reproducibility. It
contains the core code, required lightweight raw inputs, selected final summary
tables, and documentation needed to reproduce or inspect the main retained-model
analyses discussed in the manuscript.
