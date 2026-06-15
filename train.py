"""
Train a single model on the Polymarket UP-probability forecasting task.

Default model is PRISM (Polymarket-tuned config), but
any model from `model_zoo.ALL_MODELS` can be selected via `--model`.

Example
-------
    python train.py --model PRISM --epochs 15 --pred-len 1 --save-ckpt
    python train.py --model LSTM --epochs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import ALL_MODELS, build_model
from training import evaluate_polymarket, train_model

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="PRISM", choices=ALL_MODELS,
                   help="Model name; PRISM uses the Polymarket-tuned config in model_zoo.py")
    p.add_argument("--seq-len", type=int, default=96, help="History length (15m steps)")
    p.add_argument("--pred-len", type=int, default=1, help="Forecast horizon (15m steps)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--poly-only-loss", action="store_true",
                   help="Train on Polymarket channels only instead of all features")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", default=str(ROOT / "runs"))
    p.add_argument("--save-ckpt", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}  [model] {args.model}")

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size,
    )
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    print(f"[data]   channels={n_ch}  poly idx={poly_idx}  "
          f"batches train/valid/test = {len(bundle['train_loader'])}/"
          f"{len(bundle['valid_loader'])}/{len(bundle['test_loader'])}")

    model = build_model(args.model, n_ch, args.seq_len, args.pred_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]  {args.model} params={n_params/1e6:.3f}M")

    t0 = time.time()
    model, best_val = train_model(
        model, bundle["train_loader"], bundle["valid_loader"], device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, poly_idx=poly_idx,
        poly_only_loss=args.poly_only_loss,
    )
    dt = time.time() - t0
    print(f"[train]  done in {dt:.1f}s, best_val_polymse={best_val:.4f}")

    test = evaluate_polymarket(model, bundle["test_loader"], device, poly_idx,
                               scaler, ASSETS)
    print(f"\n=== TEST (Polymarket UP-prob, 0-1 space) ===")
    print(f"  MAE  avg = {test['mae_avg']:.4f}")
    print(f"  RMSE avg = {test['rmse_avg']:.4f}")
    for a in ASSETS:
        print(f"    {a.upper():4s}  MAE={test['per_asset_mae'][a]:.4f}  "
              f"RMSE={test['per_asset_rmse'][a]:.4f}")

    if args.save_ckpt:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"{args.model.lower()}_best.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_name": args.model,
            "args": vars(args),
            "columns": bundle["columns"],
            "poly_indices": poly_idx,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "test_metrics": {"mae_avg": test["mae_avg"], "rmse_avg": test["rmse_avg"]},
        }, ckpt_path)
        print(f"\n[ckpt]   saved to {ckpt_path}")


if __name__ == "__main__":
    main()
