"""Shared model + encoding utilities for Plan6 (Genome Biology style).

We keep the architecture intentionally small and CPU-friendly:
- token embedding (A/C/G/T/N)
- a few 1D convs to capture motif-like patterns
- global max pooling
- small MLP trunk
- 2 heads: mean, delta

All interpretability (PWM/ISM) is designed to focus on the delta head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
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


TOK = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
BASES = ["A", "C", "G", "T"]


def encode_seq(seq: str) -> np.ndarray:
    """Encode sequence into int tokens (A,C,G,T,N -> 0..4)."""
    return np.array([TOK.get(ch, 4) for ch in str(seq)], dtype=np.int64)


def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    s = str(seq)
    return "".join(comp.get(c, "N") for c in s[::-1])


class SmallCNNMeanDelta(nn.Module):
    """A compact 1D CNN with two regression heads (mean, delta)."""

    def __init__(self, vocab_size: int = 5, d: int = 32, dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_mean = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, delta). x_tok: (B, L)"""
        h = self.emb(x_tok).transpose(1, 2)  # (B, d, L)
        h1 = self.act(self.conv1(h))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h1))
        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h2, dim=-1)
        p3 = torch.amax(h3, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        feat = self.trunk(feat)
        mean = self.head_mean(feat).squeeze(-1)
        delta = self.head_delta(feat).squeeze(-1)
        return mean, delta


class SmallCNN3Head(nn.Module):
    """A compact 1D CNN with three regression heads: WT, MT, and Δ.

    This matches the training script train_cnn_wt_mt_delta3head.py.
    The trunk is intentionally identical to SmallCNNMeanDelta so that
    interpretability utilities (conv1 PWM, etc.) work consistently.
    """

    def __init__(self, vocab_size: int = 5, d: int = 32, dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_wt = nn.Linear(128, 1)
        self.head_mt = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor):
        h = self.emb(x_tok).transpose(1, 2)  # (B, d, L)
        h1 = self.act(self.conv1(h))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h1))
        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h2, dim=-1)
        p3 = torch.amax(h3, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        feat = self.trunk(feat)
        p_wt = self.head_wt(feat).squeeze(-1)
        p_mt = self.head_mt(feat).squeeze(-1)
        p_delta = self.head_delta(feat).squeeze(-1)
        return p_wt, p_mt, p_delta


class _ResDilatedBlock(nn.Module):
    """Residual dilated conv block (CPU-friendly).

    Keeps channel width fixed and expands receptive field via dilation.
    """

    def __init__(self, ch: int = 128, k: int = 7, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.act(self.conv2(h))
        h = self.drop(h)
        return x + h


class ResDilatedCNNMeanDelta(nn.Module):
    """A stronger CNN than SmallCNNMeanDelta using residual + dilated blocks.

    Design goal: increase receptive field and compositional capacity
    while staying CPU-friendly and keeping interpretability hooks:
    - model.emb
    - model.conv1
    """

    def __init__(
        self,
        vocab_size: int = 5,
        d: int = 32,
        dropout: float = 0.25,
        block_dropout: float = 0.12,
        dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        # keep conv1 as the first "motif" layer (used by PWM scripts)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.res_blocks = nn.ModuleList([
            _ResDilatedBlock(ch=128, k=7, dilation=int(di), dropout=block_dropout)
            for di in dilations
        ])
        self.conv_wide = nn.Conv1d(128, 128, kernel_size=13, padding=6)

        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_mean = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(x_tok).transpose(1, 2)  # (B, d, L)
        h1 = self.act(self.conv1(h))
        h = h1
        for blk in self.res_blocks:
            h = blk(h)
        h_wide = self.act(self.conv_wide(h))

        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h, dim=-1)
        p3 = torch.amax(h_wide, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        feat = self.trunk(feat)
        mean = self.head_mean(feat).squeeze(-1)
        delta = self.head_delta(feat).squeeze(-1)
        return mean, delta


class ResDilatedCNN3Head(nn.Module):
    """Residual-dilated variant of SmallCNN3Head (WT/MT/Δ)."""

    def __init__(
        self,
        vocab_size: int = 5,
        d: int = 32,
        dropout: float = 0.25,
        block_dropout: float = 0.12,
        dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.conv1 = nn.Conv1d(d, 128, kernel_size=7, padding=3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.res_blocks = nn.ModuleList([
            _ResDilatedBlock(ch=128, k=7, dilation=int(di), dropout=block_dropout)
            for di in dilations
        ])
        self.conv_wide = nn.Conv1d(128, 128, kernel_size=13, padding=6)

        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_wt = nn.Linear(128, 1)
        self.head_mt = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor):
        h = self.emb(x_tok).transpose(1, 2)
        h1 = self.act(self.conv1(h))
        h = h1
        for blk in self.res_blocks:
            h = blk(h)
        h_wide = self.act(self.conv_wide(h))

        p1 = torch.amax(h1, dim=-1)
        p2 = torch.amax(h, dim=-1)
        p3 = torch.amax(h_wide, dim=-1)
        feat = torch.cat([p1, p2, p3], dim=-1)
        feat = self.drop(feat)
        feat = self.trunk(feat)
        p_wt = self.head_wt(feat).squeeze(-1)
        p_mt = self.head_mt(feat).squeeze(-1)
        p_delta = self.head_delta(feat).squeeze(-1)
        return p_wt, p_mt, p_delta


def infer_model_kind_from_state_dict(state: dict) -> str:
    """Infer model kind from state_dict keys."""
    keys = set(state.keys())
    # Stronger residual/dilated variants have res_blocks.* keys.
    has_resblocks = any(k.startswith("res_blocks.") for k in keys)
    if {"head_mean.weight", "head_mean.bias"}.issubset(keys):
        return "mean_delta_resdilated" if has_resblocks else "mean_delta"
    if {"head_wt.weight", "head_mt.weight", "head_delta.weight"}.issubset(keys):
        return "wt_mt_delta3head_resdilated" if has_resblocks else "wt_mt_delta3head"
    # fallbacks for older naming
    if any(k.startswith("head_mean") for k in keys):
        return "mean_delta_resdilated" if has_resblocks else "mean_delta"
    if any(k.startswith("head_wt") for k in keys) and any(k.startswith("head_mt") for k in keys):
        return "wt_mt_delta3head_resdilated" if has_resblocks else "wt_mt_delta3head"
    raise ValueError(
        "Could not infer model kind from checkpoint state_dict keys. "
        "Expected heads for mean/delta or WT/MT/delta. "
        f"Example keys: {sorted(list(keys))[:12]}"
    )


def build_model_for_kind(kind: str, vocab_size: int = 5, d: int = 32, dropout: float = 0.25) -> nn.Module:
    kind = str(kind)
    if kind == "mean_delta":
        return SmallCNNMeanDelta(vocab_size=vocab_size, d=d, dropout=dropout)
    if kind == "mean_delta_resdilated":
        return ResDilatedCNNMeanDelta(vocab_size=vocab_size, d=d, dropout=dropout)
    if kind == "wt_mt_delta3head":
        return SmallCNN3Head(vocab_size=vocab_size, d=d, dropout=dropout)
    if kind == "wt_mt_delta3head_resdilated":
        return ResDilatedCNN3Head(vocab_size=vocab_size, d=d, dropout=dropout)
    raise ValueError(f"Unknown model kind: {kind}")


def load_any_cnn_checkpoint(ckpt_path: str, device: str = "cpu") -> tuple[nn.Module, str]:
    """Load either a 2-head (mean/delta) or 3-head (WT/MT/Δ) Plan8 CNN checkpoint.

    Returns (model, kind).
    """
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not look like a state_dict or {'model': state_dict}.")
    kind = infer_model_kind_from_state_dict(state)
    model = build_model_for_kind(kind)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, kind


def derive_int_epi(mean: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given mean, delta, derive INT and EPI predictions in log2 space."""
    pred_int = mean + 0.5 * delta
    pred_epi = mean - 0.5 * delta
    return pred_int, pred_epi
