"""
Drop-in wrappers for Time-Series-Library (TSLib) baselines so they share the
same `forward(x: [B, L, N]) -> [B, pred_len, N]` interface as everything else in
`model_zoo`.

Currently wired (Groups A + B from the reviewer list):
    TimeXer    (NeurIPS 2024) - exogenous-aware patched transformer
    TimeMixer  (ICLR 2024)    - multiscale all-MLP mixing
    TSMixer    (TMLR 2023)    - all-MLP with explicit channel (cross-variate) mixing
    TimesNet   (ICLR 2023)    - 2D temporal-variation blocks
    SegRNN     (2023)         - segment RNN (degenerates to per-step GRU at pred_len=1)

Mamba / S-Mamba are intentionally NOT wired here: they require `mamba_ssm`
(a CUDA-compiled package) which is not installed. Per the plan we keep SegRNN
for Group B and skip the state-space models until the env is sorted.

Isolated import
---------------
TSLib ships its own top-level `layers/`, `models/`, and `utils/` packages that
collide with other projects' same-named packages on `sys.path`. `_load_tslib`
loads each model file with the TSLib root as the *only* source for those
packages, purging and restoring `sys.modules` around the exec - the same trick
`model_zoo._load_isolated_model` uses for Autoformer/FEDformer.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_DEFAULT_TSLIB = Path(__file__).resolve().parent.parent / "baselines" / "Time-Series-Library"
TSLIB_DIR = Path(os.environ.get("TSLIB_PATH", str(_DEFAULT_TSLIB)))
if not TSLIB_DIR.exists():
    raise RuntimeError(
        f"Time-Series-Library not found at {TSLIB_DIR}. Set TSLIB_PATH or clone "
        f"https://github.com/thuml/Time-Series-Library there."
    )

_COLLIDING_PKGS = ("layers", "models", "utils")


def _load_tslib(model_filename: str, alias: str):
    """Exec a TSLib model file with TSLib as the only source for its packages."""
    file_path = TSLIB_DIR / "models" / model_filename
    saved_path = list(sys.path)
    saved_mods = {
        k: sys.modules[k] for k in list(sys.modules)
        if k.split(".")[0] in _COLLIDING_PKGS
    }
    for k in list(saved_mods):
        del sys.modules[k]
    sys.path.insert(0, str(TSLIB_DIR))
    try:
        spec = importlib.util.spec_from_file_location(alias, str(file_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Model
    finally:
        for k in list(sys.modules):
            if k.split(".")[0] in _COLLIDING_PKGS:
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# Config: TSLib models read a single flat config object. We provide every
# attribute the wired models touch, then apply per-model overrides.
# ---------------------------------------------------------------------------
def _base_config(n_channels: int, seq_len: int, pred_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="M",                 # predict all N channels (eval picks poly_idx)
        seq_len=seq_len,
        label_len=seq_len // 2,
        pred_len=pred_len,
        enc_in=n_channels,
        dec_in=n_channels,
        c_out=n_channels,
        d_model=64,
        d_ff=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        dropout=0.1,
        factor=1,
        activation="gelu",
        embed="timeF",
        freq="h",                     # -> 4 time-mark features
        use_norm=1,
        num_class=0,
        moving_avg=25,
        top_k=5,
        num_kernels=6,
    )


_N_MARKS = 4  # freq="h" time features


class _TSLibWrap(nn.Module):
    """Adapts a TSLib model to forward(x: [B, L, N]) -> [B, pred_len, N]."""

    def __init__(self, model: nn.Module, cfg: SimpleNamespace):
        super().__init__()
        self.m = model
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.label_len = cfg.label_len

    def forward(self, x):
        B, L, N = x.shape
        dev = x.device
        x_mark_enc = torch.zeros(B, L, _N_MARKS, device=dev)
        x_dec = torch.zeros(B, self.label_len + self.pred_len, N, device=dev)
        x_dec[:, : self.label_len, :] = x[:, -self.label_len:, :]
        x_mark_dec = torch.zeros(B, self.label_len + self.pred_len, _N_MARKS, device=dev)
        return self.m(x, x_mark_enc, x_dec, x_mark_dec)


# ---------------------------------------------------------------------------
# Per-model factories
# ---------------------------------------------------------------------------
def _build_timexer(cfg):
    cfg.patch_len = 16  # seq_len(96) // patch_len(16) = 6 patches
    cfg.use_norm = 1
    return _load_tslib("TimeXer.py", "tslib_timexer")(cfg)


def _build_timemixer(cfg):
    cfg.channel_independence = 1
    cfg.decomp_method = "moving_avg"
    cfg.down_sampling_layers = 2
    cfg.down_sampling_method = "avg"
    cfg.down_sampling_window = 2   # scales 96 / 48 / 24
    cfg.use_norm = 1
    return _load_tslib("TimeMixer.py", "tslib_timemixer")(cfg)


def _build_tsmixer(cfg):
    return _load_tslib("TSMixer.py", "tslib_tsmixer")(cfg)


def _build_timesnet(cfg):
    cfg.d_ff = 64       # TimesNet inception blocks are heavy; keep d_ff modest
    return _load_tslib("TimesNet.py", "tslib_timesnet")(cfg)


def _build_segrnn(cfg):
    # SegRNN needs seg_len | seq_len and seg_len | pred_len. With pred_len=1
    # the only valid choice is seg_len=1 (the model degenerates to a GRU).
    cfg.seg_len = 1 if cfg.pred_len == 1 else min(48, cfg.pred_len)
    return _load_tslib("SegRNN.py", "tslib_segrnn")(cfg)


_FACTORIES = {
    "timexer": _build_timexer,
    "timemixer": _build_timemixer,
    "tsmixer": _build_tsmixer,
    "timesnet": _build_timesnet,
    "segrnn": _build_segrnn,
}

TSLIB_MODELS = ("TimeXer", "TimeMixer", "TSMixer", "TimesNet", "SegRNN")


def build_tslib_model(name: str, n_channels: int, seq_len: int, pred_len: int) -> nn.Module:
    key = name.lower()
    if key not in _FACTORIES:
        raise ValueError(f"Unknown TSLib model: {name}")
    cfg = _base_config(n_channels, seq_len, pred_len)
    inner = _FACTORIES[key](cfg)
    return _TSLibWrap(inner, cfg)
