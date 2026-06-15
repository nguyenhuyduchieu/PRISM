"""
Faithful PyTorch reimplementations of the two Group C baselines that cannot be
used as-is:

HATS (IJCAI 2019) - upstream is TensorFlow 1.x. We reproduce its core mechanism:
    a per-node recurrent encoder followed by *hierarchical, multi-relation graph
    attention* - for each learnable relation type, every node attends over its
    neighbours (relation embedding + neighbour state + own state -> attention
    score), the per-relation neighbour representations are aggregated, then
    averaged across relations. Original HATS uses an external relation graph
    (Wikidata corporate links); with only 4 crypto assets we use R learnable
    relations over the fully-connected asset set, which keeps the attention
    mechanism faithful while fitting our data.

StockFormer (IJCAI 2023) - upstream is an RL trading agent (SAC + predictive
    coding). We reproduce its forecasting *predictive-coding core*: the
    Informer-style encoder-decoder transformer (`Transformer_base`) with a
    value+positional embedding, a self-attentive encoder over the history and a
    causal decoder that cross-attends to the encoder memory to emit the forecast.
    The upstream adds a ranking term to the loss at train time; here every model
    shares the pipeline's MSE objective for a fair comparison, so we keep the
    architecture and drop the training-recipe extras.

Both expose forward(x: [B, L, N]) -> [B, pred_len, N].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

A_ASSETS = 4
F_FEATS = 4


# ===========================================================================
# HATS
# ===========================================================================
class HATSReimpl(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len,
                 hidden=64, n_layers=1, num_relations=3, rel_dim=8, dropout=0.1):
        super().__init__()
        assert n_channels == A_ASSETS * F_FEATS
        assert pred_len == 1
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.hidden = hidden
        self.num_relations = num_relations

        # Per-node recurrent encoder (HATS uses a stacked LSTM per company)
        self.encoder = nn.LSTM(
            F_FEATS, hidden, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Learnable relation embeddings (replace Wikidata relation one-hots)
        self.rel_emb = nn.Parameter(torch.randn(num_relations, rel_dim) * 0.1)
        # Attention scorer over [neighbour_state, own_state, relation_emb]
        self.att = nn.Linear(2 * hidden + rel_dim, 1)
        # Per-node head: combine own state + relation-aggregated state -> F features
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, F_FEATS),
        )

    def _graph_attention(self, h):
        # h: [B, A, D]
        B, A, D = h.shape
        hi = h.unsqueeze(2).expand(B, A, A, D)      # query node i
        hj = h.unsqueeze(1).expand(B, A, A, D)      # neighbour node j
        # mask self-edges (j == i)
        self_mask = torch.eye(A, device=h.device, dtype=torch.bool).view(1, A, A, 1)
        rel_reps = []
        for r in range(self.num_relations):
            re = self.rel_emb[r].view(1, 1, 1, -1).expand(B, A, A, -1)
            att_x = torch.cat([hj, hi, re], dim=-1)         # [B,A,A,2D+rel]
            score = self.att(att_x)                         # [B,A,A,1]
            score = score.masked_fill(self_mask, float("-inf"))
            alpha = torch.softmax(score, dim=2)             # over neighbours j
            rel_reps.append((hj * alpha).sum(dim=2))        # [B,A,D]
        updated = torch.stack(rel_reps, dim=0).mean(dim=0)  # mean over relations
        return updated

    def forward(self, x):
        B, L, N = x.shape
        g = x.view(B, L, A_ASSETS, F_FEATS).permute(0, 2, 1, 3)  # [B,A,L,F]
        ba = g.reshape(B * A_ASSETS, L, F_FEATS)
        _, (hn, _) = self.encoder(ba)
        h = hn[-1].view(B, A_ASSETS, self.hidden)               # [B,A,D]
        updated = self._graph_attention(h)                      # [B,A,D]
        out = self.head(torch.cat([h, updated], dim=-1))        # [B,A,F]
        return out.reshape(B, self.pred_len, N)


# ===========================================================================
# StockFormer (Transformer_base predictive-coding core)
# ===========================================================================
class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(c_in, d_model)
        self.pos = _PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.value(x) + self.pos(x))


class StockFormerReimpl(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = seq_len // 2
        self.n_channels = n_channels

        self.enc_embedding = _DataEmbedding(n_channels, d_model, dropout)
        self.dec_embedding = _DataEmbedding(n_channels, d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=d_layers)
        self.projection = nn.Linear(d_model, n_channels)

    def forward(self, x):
        B, L, N = x.shape
        # decoder input: last label_len of history + zeros for the horizon
        dec_in = torch.zeros(B, self.label_len + self.pred_len, N, device=x.device)
        dec_in[:, : self.label_len, :] = x[:, -self.label_len:, :]

        enc_out = self.encoder(self.enc_embedding(x))
        T = self.label_len + self.pred_len
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        dec_out = self.decoder(self.dec_embedding(dec_in), enc_out, tgt_mask=causal)
        out = self.projection(dec_out)
        return out[:, -self.pred_len:, :]


GROUPC_REIMPL_MODELS = ("HATS", "StockFormer")


def build_groupc_reimpl(name, n_channels, seq_len, pred_len):
    key = name.lower()
    if key == "hats":
        return HATSReimpl(n_channels, seq_len, pred_len)
    if key == "stockformer":
        return StockFormerReimpl(n_channels, seq_len, pred_len)
    raise ValueError(f"Unknown Group C reimpl model: {name}")
