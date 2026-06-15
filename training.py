"""
Shared train / eval helpers used by `train.py` and `benchmark.py`.

All evaluators measure MAE / RMSE on the *Polymarket UP-probability channels
only* and report the result in original probability space (0-1), after
inverse-standardization with the loader's saved scaler.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    patience: int = 6,
    grad_clip: float = 1.0,
    poly_idx: Optional[List[int]] = None,
    poly_only_loss: bool = False,
    max_train_batches: int = 0,
) -> Tuple[nn.Module, float]:
    """Train with AdamW + cosine LR schedule + early stopping on validation MSE.

    If `poly_only_loss=True` the training MSE is computed only on the channels
    in `poly_idx` (the Polymarket UP-prob targets). Validation MSE is always
    computed on `poly_idx` if provided, otherwise on all channels.

    Returns (model with best_state loaded, best validation MSE).
    """
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            if max_train_batches and bi >= max_train_batches:
                break
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            pred = model(x)
            if pred.shape != y.shape:
                pred = pred.view_as(y)
            if poly_only_loss and poly_idx is not None:
                loss = loss_fn(pred[..., poly_idx], y[..., poly_idx])
            else:
                loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vs = 0.0; vn = 0
            for x, y in valid_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x)
                if pred.shape != y.shape:
                    pred = pred.view_as(y)
                if poly_idx is not None:
                    vs += float(((pred[..., poly_idx] - y[..., poly_idx]) ** 2).mean().item())
                else:
                    vs += float(((pred - y) ** 2).mean().item())
                vn += 1
            vloss = vs / max(vn, 1)

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val


def evaluate_polymarket(
    model: nn.Module,
    loader,
    device: torch.device,
    poly_idx: List[int],
    scaler,
    assets: List[str],
) -> Dict:
    """Per-asset MAE / RMSE on Polymarket UP-prob channels (original 0-1 space)."""
    model.eval()
    poly_mu = scaler.mean[poly_idx]
    poly_sd = scaler.std[poly_idx]
    sse = np.zeros(len(poly_idx))
    sae = np.zeros(len(poly_idx))
    n = np.zeros(len(poly_idx))
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            if pred.shape != y.shape:
                pred = pred.view_as(y)
            pp = pred[..., poly_idx].cpu().numpy() * poly_sd + poly_mu
            yp = y[..., poly_idx].cpu().numpy() * poly_sd + poly_mu
            d = pp - yp
            sse += (d ** 2).sum(axis=(0, 1))
            sae += np.abs(d).sum(axis=(0, 1))
            n += d.shape[0] * d.shape[1]
    mae = sae / n
    rmse = np.sqrt(sse / n)
    return {
        "mae": mae, "rmse": rmse,
        "mae_avg": float(mae.mean()), "rmse_avg": float(rmse.mean()),
        "per_asset_mae": {a: float(v) for a, v in zip(assets, mae)},
        "per_asset_rmse": {a: float(v) for a, v in zip(assets, rmse)},
    }


def evaluate_naive(loader, poly_idx: List[int], scaler) -> Dict[str, Dict]:
    """Persistence (`y_hat = y[t-1]`) and Mean=0.5 baselines."""
    poly_mu = scaler.mean[poly_idx]
    poly_sd = scaler.std[poly_idx]
    sae_p = np.zeros(len(poly_idx)); sse_p = np.zeros(len(poly_idx))
    sae_m = np.zeros(len(poly_idx)); sse_m = np.zeros(len(poly_idx))
    n = np.zeros(len(poly_idx))
    for x, y in loader:
        yp = y[..., poly_idx].numpy() * poly_sd + poly_mu
        last = x[:, -1, poly_idx].numpy() * poly_sd + poly_mu
        pred_pers = np.repeat(last[:, None, :], yp.shape[1], axis=1)
        pred_mean = np.full_like(yp, 0.5)
        d1 = pred_pers - yp; d2 = pred_mean - yp
        sae_p += np.abs(d1).sum(axis=(0, 1)); sse_p += (d1 ** 2).sum(axis=(0, 1))
        sae_m += np.abs(d2).sum(axis=(0, 1)); sse_m += (d2 ** 2).sum(axis=(0, 1))
        n += d1.shape[0] * d1.shape[1]
    return {
        "Persistence": {
            "mae": sae_p / n, "rmse": np.sqrt(sse_p / n),
            "mae_avg": float((sae_p / n).mean()),
            "rmse_avg": float(np.sqrt(sse_p / n).mean()),
        },
        "Mean=0.5": {
            "mae": sae_m / n, "rmse": np.sqrt(sse_m / n),
            "mae_avg": float((sae_m / n).mean()),
            "rmse_avg": float(np.sqrt(sse_m / n).mean()),
        },
    }
