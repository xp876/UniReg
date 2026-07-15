# GitHub Upload Checklist

Use this directory as the GitHub-ready repository root.

## Required files present

- `README.md`
- `LICENSE`
- `environment.yml`
- `requirements.txt`
- `DATA_AVAILABILITY.md`
- `Data/raw_manifest.tsv`
- `Docs/Unireg.pdf`
- `Docs/Unireg.png`
- `Docs/unireg-v3.pdf`
- `Docs/MODEL_ARCHITECTURE.md`
- `Code/`
- `Data/raw/GSE83894/`
- `Data/raw/GSE142696/`
- `Results/`
- `Supplementary_Material/`

## Raw inputs included

- `Data/raw/GSE83894/formatB_agg_only.zip`
- `Data/raw/GSE83894/formatA_all_replicates.zip`
- `Data/raw/GSE142696/GSM4237954_9MPRA_elements.fa.gz`
- `Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.tsv.gz`
- `Data/raw/GSE142696/GSE142696_9MPRA.ActivityRatios.IndividualReps.tsv.gz`

## Important notes

- The repository is compact and manuscript-focused.
- `Docs/Unireg.pdf` and `Docs/Unireg.png` were regenerated from the latest
  provided v3 schematic.
- Full checkpoints, complete per-element predictions and large intermediate
  work directories are intentionally excluded.
- The optional motif database `H14CORE_meme_format.meme` is not included.
- `gkmsvm_optional` and `nt_transformer_delta_ens` require external resources.
- Python bytecode caches (`__pycache__/`) were removed from the clean upload
  directory.
