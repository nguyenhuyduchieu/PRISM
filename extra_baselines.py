"""
Input-conditioning control baselines requested by a reviewer.

The reviewer's concern: PRISM's edge over the fixed-linear field might come from
generic *input-conditioning* rather than from PRISM's specific architecture
(regime encoder + dynamic graph + multi-scale frequency bank + hyper-linear). To
isolate this we add two minimal input-conditioned linear models that share the
fairness scaffolding of the other baselines (RevIN, channel-shared weights,
channel-individual application -- no cross-channel mixing):

  * MoLE  -- Mixture-of-Linear-Experts (Ni et al., 2024): K linear experts mapping
    the look-back to the horizon, combined by input-conditioned softmax gates.
  * AdaLinear -- a single RevIN linear core whose weight receives an
    input-conditioned low-rank correction (PRISM's hyper-linear with the regime
    encoder, dynamic graph, and frequency bank removed). This is the tightest
    "input-conditioning, nothing else" control.

Both expose forward(x: [B, L, N]) -> [B, pred_len, N].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        else:  # denorm
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x


class MoLE(nn.Module):
    """Mixture-of-Linear-Experts: K shared linear maps L->H combined by an
    input-conditioned softmax gate. Weights are shared across channels and each
    channel is forecast from its own history (no cross-channel mixing)."""

    def __init__(self, n_channels, seq_len, pred_len, num_experts=4,
                 gate_hidden=64):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.K = num_experts
        self.revin = _RevIN(n_channels)

        # K experts: weight [K, H, L], bias [K, H]
        self.W = nn.Parameter(torch.empty(num_experts, pred_len, seq_len))
        nn.init.xavier_uniform_(self.W, gain=0.1)
        self.b = nn.Parameter(torch.zeros(num_experts, pred_len))

        # input-conditioned gate over a global (channel-averaged) window summary
        self.gate = nn.Sequential(
            nn.Linear(seq_len, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )

    def forward(self, x):
        # x: [B, L, N]
        xn = self.revin(x, "norm")
        ctx = xn.mean(dim=2)                       # [B, L] channel-averaged window
        g = F.softmax(self.gate(ctx), dim=-1)      # [B, K]
        # experts: [B, K, H, N]
        experts = torch.einsum("khl,bln->bkhn", self.W, xn) + self.b[None, :, :, None]
        y = (g[:, :, None, None] * experts).sum(dim=1)   # [B, H, N]
        return self.revin(y, "denorm")


class AdaLinear(nn.Module):
    """RevIN linear core + input-conditioned low-rank weight correction.

    This is PRISM's hyper-linear in isolation: the regime encoder, dynamic graph,
    and frequency bank are removed, so the only adaptivity is a low-rank update to
    the shared forecasting weight generated from a global window summary. Channels
    are forecast independently with the shared (per-sample) weight."""

    def __init__(self, n_channels, seq_len, pred_len, rank=4, ctx_hidden=64):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rank = rank
        self.revin = _RevIN(n_channels)

        self.W_base = nn.Parameter(torch.empty(pred_len, seq_len))
        nn.init.xavier_uniform_(self.W_base, gain=0.1)

        self.hyper = nn.Sequential(
            nn.Linear(seq_len, ctx_hidden),
            nn.LayerNorm(ctx_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ctx_hidden, rank * (seq_len + pred_len)),
        )
        self.adapt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L, N = x.shape
        xn = self.revin(x, "norm")
        ctx = xn.mean(dim=2)                       # [B, L]
        h = self.hyper(ctx)
        A = h[:, : self.rank * self.pred_len].view(B, self.pred_len, self.rank)
        Bd = h[:, self.rank * self.pred_len:].view(B, self.rank, self.seq_len)
        W = self.W_base[None] + self.adapt_scale * (A @ Bd)   # [B, H, L]
        y = torch.einsum("bhl,bln->bhn", W, xn)               # [B, H, N]
        return self.revin(y, "denorm")


def build_extra_model(name, n_channels, seq_len, pred_len):
    n = name.lower()
    if n == "mole":
        return MoLE(n_channels, seq_len, pred_len)
    if n == "adalinear":
        return AdaLinear(n_channels, seq_len, pred_len)
    raise ValueError(f"Unknown extra baseline: {name}")


EXTRA_MODELS = ("MoLE", "AdaLinear")
