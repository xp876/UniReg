import argparse
from pathlib import Path
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load correct CNN architecture automatically based on checkpoint keys.
from models import load_any_cnn_checkpoint

BASES = ["A","C","G","T","N"]
BASE_IDX = {b:i for i,b in enumerate(BASES)}

def softmax(x: np.ndarray, axis: int=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to cnn_mu_delta_best.pt or similar")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_filters", type=int, default=16, help="How many filters to plot as heatmaps")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, kind = load_any_cnn_checkpoint(args.ckpt, device="cpu")
    # kind is one of: mean_delta, wt_mt_delta3head
    print(f"[explain_conv1_pwm] loaded kind={kind} ckpt={args.ckpt}")

    # Extract embedding and conv1 weights
    emb = model.emb.weight.detach().cpu().numpy()  # (5, d)
    W = model.conv1.weight.detach().cpu().numpy()  # (out=128, in=d, k)

    out_ch, d, k = W.shape
    eff = np.zeros((out_ch, k, len(BASES)), dtype=np.float32)
    for oc in range(out_ch):
        for j in range(k):
            w_vec = W[oc, :, j]  # (d,)
            # dot with each base embedding => scalar
            for bi in range(len(BASES)):
                eff[oc, j, bi] = float(np.dot(w_vec, emb[bi]))

    # Convert to PWM-like probabilities for A/C/G/T only (ignore N)
    eff_acgt = eff[:, :, :4]  # (F, k, 4)
    pwm = softmax(eff_acgt, axis=-1)  # per-position categorical
    # info content per position (bits): 2 - H(p)
    H = -np.sum(pwm * np.log2(np.clip(pwm, 1e-8, 1.0)), axis=-1)
    ic = 2.0 - H  # (F, k)

    # Save long CSV
    rows = []
    for f in range(out_ch):
        for j in range(k):
            for bi, b in enumerate(["A","C","G","T"]):
                rows.append({
                    "filter": f,
                    "pos": j,
                    "base": b,
                    "weight": float(eff[f, j, bi]),
                    "prob": float(pwm[f, j, bi]),
                    "ic_pos": float(ic[f, j]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "conv1_effective_pwm.tsv", sep="\t", index=False)

    np.savez(out_dir / "conv1_effective_pwm.npz", eff=eff, pwm=pwm, ic=ic)
    # Heatmap plots for first N filters (or pick by mean IC)
    order = np.argsort(-ic.mean(axis=1))
    maxF = min(int(args.max_filters), out_ch)
    for rank in range(maxF):
        f = int(order[rank])
        mat = pwm[f].T  # (4, k)
        plt.figure(figsize=(k/2.5, 2.0))
        plt.imshow(mat, aspect="auto")
        plt.yticks([0,1,2,3], ["A","C","G","T"])
        plt.xticks(range(k))
        plt.colorbar(label="prob")
        plt.title(f"conv1 filter {f} (meanIC={ic[f].mean():.2f} bits)")
        plt.tight_layout()
        plt.savefig(out_dir / f"conv1_filter{f:03d}_heatmap.png", dpi=200)
        plt.close()

    print("Wrote:", out_dir)

if __name__ == "__main__":
    main()
