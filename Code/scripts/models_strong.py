"""Stronger CPU-friendly sequence models for Plan8 NatMethods Analysis upgrade.

This module adds a multi-scale residual dilated CNN that provides a stronger
inductive bias for long-range motif interactions than the baseline compact CNN.

Design goals:
- CPU-friendly (no huge channel counts)
- Multi-scale receptive fields via dilation
- Residual connections for stability
- Three-head outputs (WT / MT / Delta)

The training script `train_cnn_wt_mt_delta3head_msres.py` uses this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResDilatedBlock(nn.Module):
    def __init__(self, c: int, k: int = 7, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(c)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class MultiScaleResCNN3Head(nn.Module):
    """Multi-scale residual dilated CNN trunk with WT/MT/Δ heads."""

    def __init__(
        self,
        vocab_size: int = 5,
        emb_d: int = 32,
        trunk_c: int = 128,
        blocks_per_scale: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_d)
        self.proj = nn.Conv1d(emb_d, trunk_c, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        towers = []
        for d in dilations:
            layers = []
            for _ in range(int(blocks_per_scale)):
                layers.append(ResDilatedBlock(trunk_c, k=7, dilation=int(d), dropout=dropout * 0.4))
            towers.append(nn.Sequential(*layers))
        self.towers = nn.ModuleList(towers)

        feat_dim = trunk_c * len(dilations)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_wt = nn.Linear(128, 1)
        self.head_mt = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x_tok: torch.Tensor):
        # x_tok: (B, L) int tokens
        h = self.emb(x_tok).transpose(1, 2)  # (B, emb_d, L)
        h = self.act(self.proj(h))  # (B, C, L)

        pooled = []
        for tower in self.towers:
            t = tower(h)
            pooled.append(torch.amax(t, dim=-1))
        feat = torch.cat(pooled, dim=-1)
        feat = self.drop(feat)
        feat = self.mlp(feat)

        p_wt = self.head_wt(feat).squeeze(-1)
        p_mt = self.head_mt(feat).squeeze(-1)
        p_delta = self.head_delta(feat).squeeze(-1)
        return p_wt, p_mt, p_delta
