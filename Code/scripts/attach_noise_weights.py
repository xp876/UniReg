import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True, help="Directory with train.tsv/val.tsv/test.tsv")
    ap.add_argument("--weights_tsv", required=True, help="Per-element weights table from compute_noise_weights_from_formatA.py")
    ap.add_argument("--out_dir", required=True, help="Output prepared_dir with weights columns")
    ap.add_argument("--keep_existing", action="store_true", help="If set, keep existing w_* columns and only fill missing")
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv(args.weights_tsv, sep="\t")
    w["element_id"] = w["element_id"].astype(str)

    for sp in ["train", "val", "test"]:
        df = pd.read_csv(prep / f"{sp}.tsv", sep="\t")
        df["element_id"] = df["element_id"].astype(str)
        merged = df.merge(w[["element_id", "w_int", "w_epi", "w_mean", "w_delta"]], on="element_id", how="left")
        # fallback to sample_weight if present
        if "sample_weight" in merged.columns:
            for c in ["w_int", "w_epi", "w_mean", "w_delta"]:
                if c not in merged.columns:
                    merged[c] = merged["sample_weight"].astype(float)
                else:
                    merged[c] = merged[c].fillna(merged["sample_weight"].astype(float))
        else:
            for c in ["w_int", "w_epi", "w_mean", "w_delta"]:
                if c not in merged.columns:
                    merged[c] = 1.0
                merged[c] = merged[c].fillna(1.0)

        merged.to_csv(out / f"{sp}.tsv", sep="\t", index=False)

    # copy qc json if present
    qc = prep / "qc_paired.json"
    if qc.exists():
        (out / "qc_paired.json").write_text(qc.read_text(encoding="utf-8"), encoding="utf-8")

    print("Wrote weighted prepared dataset to:", out)


if __name__ == "__main__":
    main()
