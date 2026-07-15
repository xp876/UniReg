import argparse
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "ERROR: PyTorch (torch) is not installed.\n"
        "This project requires PyTorch even on CPU.\n\n"
        "Install (recommended, conda):\n"
        "  conda install -c pytorch pytorch cpuonly -y\n\n"
        "Or install with pip (CPU wheels):\n"
        "  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
    ) from e

import matplotlib.pyplot as plt

from common import load_prepared_split
from models import encode_seq, load_any_cnn_checkpoint

BASES = ["A", "C", "G", "T"]


def element_prefix(eid: str) -> str:
    s = str(eid)
    if ":" in s:
        return s.split(":", 1)[0]
    return "UNK"


def _safe_stem(raw: str, max_prefix: int = 60) -> str:
    raw = str(raw)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if len(prefix) > max_prefix:
        prefix = prefix[:max_prefix]
    if not prefix:
        prefix = "elem"
    return f"{prefix}__{h}"


def predict_delta(model: torch.nn.Module, seqs: list[str]) -> np.ndarray:
    """Return Δ predictions regardless of whether the model is 2-head or 3-head."""
    xs = [torch.from_numpy(encode_seq(s)) for s in seqs]
    x = torch.stack(xs, dim=0)
    with torch.no_grad():
        out = model(x)
    # out can be (mean, delta) or (wt, mt, delta)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        delta = out[1]
    elif isinstance(out, (tuple, list)) and len(out) == 3:
        delta = out[2]
    else:
        raise RuntimeError(f"Unexpected model forward() output type/len: {type(out)}")
    return delta.detach().cpu().numpy()


def load_pick_list(args) -> list[str] | None:
    ids = []
    if args.element_ids.strip():
        ids.extend([x.strip() for x in args.element_ids.split(",") if x.strip()])
    if args.element_ids_tsv.strip():
        df = pd.read_csv(args.element_ids_tsv, sep=None, engine="python")
        col = "element_id" if "element_id" in df.columns else df.columns[0]
        ids.extend(df[col].astype(str).tolist())
    ids = [x for x in ids if x]
    return ids if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])

    ap.add_argument("--top_pos", type=int, default=5, help="Number of highest +delta elements")
    ap.add_argument("--top_neg", type=int, default=5, help="Number of lowest -delta elements")
    ap.add_argument("--only_controls", action="store_true", help="Restrict picks to controls (element_id starts with 'C:')")
    ap.add_argument("--prefix", default="", help="If set, restrict picks to this element_id prefix, e.g. R")

    ap.add_argument("--element_ids", default="", help="Comma-separated element_ids to run ISM on")
    ap.add_argument("--element_ids_tsv", default="", help="TSV/CSV containing element_id column")

    ap.add_argument("--max_len", type=int, default=171)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_prepared_split(str(Path(args.prepared_dir) / f"{args.split}.tsv"))
    if "delta" not in df.columns:
        df["delta"] = pd.to_numeric(df["log2_WT"], errors="coerce") - pd.to_numeric(df["log2_MT"], errors="coerce")

    df = df.dropna(subset=["sequence", "delta"]).copy()
    df["_pref"] = df["element_id"].astype(str).map(element_prefix)

    if args.only_controls:
        df = df[df["element_id"].astype(str).str.startswith("C:")].copy()

    if args.prefix.strip():
        df = df[df["_pref"] == args.prefix.strip()].copy()

    pick = load_pick_list(args)
    if pick is not None:
        picked = df[df["element_id"].astype(str).isin(set(pick))].copy()
        if picked.empty:
            raise SystemExit("No requested element_ids found in the selected split")
    else:
        pos = df.sort_values("delta", ascending=False).head(int(args.top_pos)).copy()
        neg = df.sort_values("delta", ascending=True).head(int(args.top_neg)).copy()
        picked = pd.concat([pos, neg], ignore_index=True)

    model, kind = load_any_cnn_checkpoint(args.ckpt, device="cpu")
    print(f"[ism_delta_gb] loaded kind={kind} ckpt={args.ckpt}")

    all_imp_rows = []
    filemap_rows = []

    for _, row in picked.iterrows():
        eid = str(row["element_id"])
        seq = str(row["sequence"])[: int(args.max_len)]
        L = len(seq)

        ref_pred = float(predict_delta(model, [seq])[0])
        true_delta = float(row["delta"])

        records = []
        for i in range(L):
            ref = seq[i]
            if ref not in BASES:
                continue
            for b in BASES:
                if b == ref:
                    continue
                mut_seq = seq[:i] + b + seq[i + 1 :]
                mut_pred = float(predict_delta(model, [mut_seq])[0])
                records.append(
                    {
                        "element_id": eid,
                        "pos": i,
                        "ref": ref,
                        "mut": b,
                        "pred_delta_ref": ref_pred,
                        "pred_delta_mut": mut_pred,
                        "diff": mut_pred - ref_pred,
                    }
                )

        rec_df = pd.DataFrame(records)
        safe = _safe_stem(eid)
        out_path = out_dir / f"ism_{safe}.tsv"
        rec_df.to_csv(out_path, sep="\t", index=False)

        if len(rec_df) == 0:
            continue

        imp = (
            rec_df.groupby("pos")["diff"]
            .apply(lambda x: float(np.max(np.abs(x.values))))
            .reset_index()
            .sort_values("pos")
            .rename(columns={"diff": "importance"})
        )
        imp["element_id"] = eid
        imp["true_delta"] = true_delta
        imp["pred_delta_ref"] = ref_pred
        imp.to_csv(out_dir / f"ism_{safe}.importance.tsv", sep="\t", index=False)

        plt.figure(figsize=(8, 3))
        plt.plot(imp["pos"].values, imp["importance"].values)
        plt.title(f"ISM on Δ head\ntrue Δ={true_delta:.3f}, pred Δ={ref_pred:.3f}\n{eid}")
        plt.xlabel("position")
        plt.ylabel("max |Δ_pred(mut)-Δ_pred(ref)|")
        plt.tight_layout()
        plt.savefig(out_dir / f"ism_{safe}.png", dpi=200)
        plt.close()

        all_imp_rows.append(imp)
        filemap_rows.append({"safe_name": safe, "element_id": eid, "path": out_path.name})

    if all_imp_rows:
        pd.concat(all_imp_rows, ignore_index=True).to_csv(out_dir / "ism_summary_all.tsv", sep="\t", index=False)
    if filemap_rows:
        pd.DataFrame(filemap_rows).to_csv(out_dir / "ism_filemap.tsv", sep="\t", index=False)

    print("Wrote ISM outputs to:", out_dir)


if __name__ == "__main__":
    main()
