# Manual GitHub Upload Guide

Target repository:

```text
https://github.com/xp876/UniReg
```

This directory is intended to be uploaded as the repository root.

## Recommended web-only upload workflow

1. Open the GitHub repository page in your browser.
2. If the repository is empty, choose **uploading an existing file**.
3. Drag the contents of this directory into the upload area, including:
   - `README.md`
   - `LICENSE`
   - `Code/`
   - `Data/`
   - `Docs/`
   - `Results/`
   - `Supplementary_Material/`
   - `DATA_AVAILABILITY.md`
   - `GITHUB_UPLOAD_CHECKLIST.md`
   - `REPOSITORY_QA_REPORT.md`
   - `environment.yml`
   - `requirements.txt`
4. Use a commit message such as:

```text
Initial UniReg manuscript reproducibility release
```

5. Commit directly to the `main` branch.

## After upload

Check the GitHub page and confirm that:

- the README renders automatically on the repository homepage;
- `Docs/Unireg.png` appears inside the README;
- the raw data files are present under `Data/raw/`;
- the code files are visible under `Code/scripts/`;
- the license badge/section shows MIT.

## Large-file note

This compact release is small enough for normal GitHub upload. It intentionally
excludes full model checkpoints, full prediction matrices and large intermediate
work directories.
