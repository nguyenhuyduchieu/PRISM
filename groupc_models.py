"""
Domain-specific (stock/crypto) baselines from the reviewer's Group C, adapted to
the Polymarket pipeline's `forward(x: [B, L, N]) -> [B, pred_len, N]` contract.

Both wrapped models are cross-sectional / cross-asset by design, so we reshape
our flat 16-channel frame into an explicit (asset x feature) grid:

    x: [B, L, 16]  ->  [B, L, A=4, F=4]      (channel j -> asset j//4, feature j%4)

The asset-major layout means the final [B, A, F] prediction flattens back to the
original 16-channel order, and the four poly_up targets stay at indices 0,4,8,12.

Wired
-----
StockMixer (AAAI 2024) - all-MLP time/channel mixing + a cross-stock mixer.
    Original predicts one scalar return per stock with no batch axis and a fixed
    lookback of 16 (conv stride-2 -> scale_dim=8). We make it batch-aware, set
    scale_dim = L//2 for our L=96, and let the two output heads emit all F
    features per asset so the full 16-channel frame is produced.

SAMBA / Graph-Mamba (ICASSP 2025) - bidirectional pure-PyTorch Mamba + a
    learnable Chebyshev graph conv over the channels. Original collapses all
    nodes to a single target series; we keep the per-node output [B, N, 1] so it
    predicts every channel, and make the graph support device-agnostic (the
    upstream hard-codes `.cuda()`).

HATS and StockFormer are NOT wired here: HATS ships only TensorFlow 1.x code and
StockFormer is an RL trading agent (SAC + predictive coding), neither of which
maps onto a plain point-forecast baseline without a from-scratch reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from groupc.stockmixer_model import MultTime2dMixer
from groupc.samba.samba import SAMBA
from groupc.samba.model_config import ModelArgs

A_ASSETS = 4
F_FEATS = 4


def _to_grid(x):
    """[B, L, A*F] -> [B, L, A, F] (asset-major)."""
    B, L, N = x.shape
    return x.view(B, L, A_ASSETS, F_FEATS)


# ---------------------------------------------------------------------------
# StockMixer adapter
# ---------------------------------------------------------------------------
class _BatchCrossStockMixer(nn.Module):
    """NoGraphMixer made batch-aware: mixes across the asset axis per sample."""

    def __init__(self, n_assets, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(n_assets)
        self.fc1 = nn.Linear(n_assets, hidden)
        self.act = nn.Hardswish()
        self.fc2 = nn.Linear(hidden, n_assets)

    def forward(self, x):  # x: [B, A, T]
        h = x.transpose(1, 2)           # [B, T, A]
        h = self.ln(h)
        h = self.fc2(self.act(self.fc1(h)))
        return h.transpose(1, 2)        # [B, A, T]


class StockMixerAdapter(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len, market=20):
        super().__init__()
        assert n_channels == A_ASSETS * F_FEATS, "StockMixer adapter expects 4x4 grid"
        assert pred_len == 1, "StockMixer adapter is single-step"
        self.pred_len = pred_len
        T = seq_len
        scale_dim = T // 2
        self.conv = nn.Conv1d(F_FEATS, F_FEATS, kernel_size=2, stride=2)
        self.mixer = MultTime2dMixer(T, F_FEATS, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(F_FEATS, 1)
        t2 = T * 2 + scale_dim
        self.time_fc = nn.Linear(t2, F_FEATS)
        self.stock_mixer = _BatchCrossStockMixer(A_ASSETS, market)
        self.time_fc_ = nn.Linear(t2, F_FEATS)

    def forward(self, x):
        B, L, N = x.shape
        g = _to_grid(x)                              # [B, L, A, F]
        ba = g.permute(0, 2, 1, 3).reshape(B * A_ASSETS, L, F_FEATS)  # [B*A, L, F]

        c = self.conv(ba.permute(0, 2, 1)).permute(0, 2, 1)          # [B*A, L/2, F]
        y = self.mixer(ba, c)                        # [B*A, t2, F]
        y = self.channel_fc(y).squeeze(-1)           # [B*A, t2]
        y = y.view(B, A_ASSETS, -1)                  # [B, A, t2]

        main = self.time_fc(y)                       # [B, A, F]
        z = self.stock_mixer(y)                      # [B, A, t2]
        z = self.time_fc_(z)                         # [B, A, F]
        out = main + z                               # [B, A, F]
        return out.reshape(B, self.pred_len, N)      # [B, 1, 16]


# ---------------------------------------------------------------------------
# SAMBA / Graph-Mamba adapter
# ---------------------------------------------------------------------------
class SambaAdapter(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len,
                 d_model=32, n_layer=2, embed=10, cheb_k=3, hid=32):
        super().__init__()
        assert pred_len == 1, "SAMBA adapter is single-step"
        self.pred_len = pred_len
        self.n_channels = n_channels
        args = ModelArgs(
            d_model=d_model, n_layer=n_layer, vocab_size=n_channels,
            seq_in=seq_len, seq_out=pred_len,
        )
        # inp = seq_in (graph conv mixes over the L-length feature axis), out = 1
        self.net = SAMBA(args, hidden=hid, inp=seq_len, out=1, embed=embed, cheb_k=cheb_k)
        nn.init.xavier_uniform_(self.net.weights_pool)
        nn.init.zeros_(self.net.bias_pool)

    def forward(self, x):
        # x: [B, L, N]. Reproduce SAMBA's graph conv but keep the per-node output
        # and stay on x.device (upstream hard-codes .cuda()).
        net = self.net
        xx = net.mam1(x)                                   # [B, L, N]
        ADJ = net.gaussian_kernel_graph(net.adj, xx, gamma=net.gamma)  # [N, N]
        I = torch.eye(x.size(2), device=x.device)
        support_set = [I, ADJ]
        for k in range(2, net.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)         # [cheb_k, N, N]
        weights = torch.einsum('nd,dkio->nkio', net.adj, net.weights_pool)
        bias = torch.matmul(net.adj, net.bias_pool)
        x_g = torch.einsum('knm,bmc->bknc', supports, xx.permute(0, 2, 1))
        x_g = x_g.permute(0, 2, 1, 3)
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, 1]
        return out.permute(0, 2, 1).reshape(x.size(0), self.pred_len, self.n_channels)


GROUPC_MODELS = ("StockMixer", "SAMBA")


def build_groupc_model(name, n_channels, seq_len, pred_len):
    key = name.lower()
    if key == "stockmixer":
        return StockMixerAdapter(n_channels, seq_len, pred_len)
    if key in ("samba", "graphmamba", "graph-mamba"):
        return SambaAdapter(n_channels, seq_len, pred_len)
    raise ValueError(f"Unknown Group C model: {name}")
