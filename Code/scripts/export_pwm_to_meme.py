import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pwm_tsv", required=True, help="conv1_effective_pwm.tsv")
    ap.add_argument("--out_meme", required=True)
    ap.add_argument("--alphabet", default="ACGT")
    ap.add_argument("--name_prefix", default="CNN_filter")
    args = ap.parse_args()

    df = pd.read_csv(args.pwm_tsv, sep="\t")
    req = {"filter", "pos", "base", "prob"}
    if not req.issubset(df.columns):
        raise SystemExit(f"PWM TSV missing columns {sorted(req)}; got {df.columns.tolist()}")

    alphabet = list(args.alphabet)
    out_path = Path(args.out_meme)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("MEME version 4\n\n")
        f.write(f"ALPHABET= {''.join(alphabet)}\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        # uniform background
        bg = 1.0 / len(alphabet)
        f.write(" ".join([f"{b} {bg:.4f}" for b in alphabet]) + "\n\n")

        for filt, sub in df.groupby("filter"):
            sub = sub.copy()
            # ensure order
            sub["pos"] = sub["pos"].astype(int)
            sub = sub.sort_values(["pos", "base"])
            positions = sorted(sub["pos"].unique().tolist())
            w = len(positions)
            f.write(f"MOTIF {args.name_prefix}{int(filt)}\n")
            f.write(f"letter-probability matrix: alength= {len(alphabet)} w= {w} nsites= {w} E= 0\n")
            for p in positions:
                row = sub[sub["pos"] == p].set_index("base")["prob"].to_dict()
                vals = [float(row.get(b, 0.0)) for b in alphabet]
                s = sum(vals)
                if s <= 0:
                    vals = [1.0 / len(alphabet)] * len(alphabet)
                else:
                    vals = [v / s for v in vals]
                f.write(" ".join([f"{v:.6f}" for v in vals]) + "\n")
            f.write("\n")

    print("Wrote MEME motifs to:", out_path)


if __name__ == "__main__":
    main()
