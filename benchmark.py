"""
Unified benchmark: train PRISM + 8 baselines + LSTM + Transformer on the
Polymarket UP-probability forecasting task and report comparable test metrics.

Default model set: all 11 in `model_zoo.ALL_MODELS`. Subsets via `--models`.

Naive baselines (Persistence, Mean=0.5) are evaluated for sanity reference.

Example
-------
    python benchmark.py --epochs 15 --pred-len 1
    python benchmark.py --models "PRISM,LSTM,Transformer"
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import ALL_MODELS, build_model
from training import evaluate_naive, evaluate_polymarket, train_model

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Default LR (PRISM overrides to 8e-4)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-csv", default=str(ROOT / "runs" / "benchmark_results_no_leak.csv"),
                   help="Output CSV path. Default overwrites the canonical no-leak results.")
    p.add_argument("--models", default=",".join(ALL_MODELS),
                   help="Comma-separated model names")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help=">0 to limit train batches per epoch (debug)")
    return p.parse_args()


def fmt_naive(name, m):
    return (f"  {name:<14}  MAE={m['mae_avg']:.4f}  RMSE={m['rmse_avg']:.4f}  "
            + "(" + " ".join(f"{a}={v:.3f}" for a, v in zip(ASSETS, m["mae"])) + ")")


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}")

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size,
    )
    n_ch = bundle["n_channels"]; poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    print(f"[data]   {len(bundle['df'])} rows, {n_ch} channels, "
          f"poly idx={poly_idx}  "
          f"range {bundle['df'].index.min()} .. {bundle['df'].index.max()}")
    print(f"[data]   batches train/valid/test = "
          f"{len(bundle['train_loader'])}/{len(bundle['valid_loader'])}/{len(bundle['test_loader'])}")

    rows = []
    print("\n=== Naive baselines (test split) ===")
    for name, m in evaluate_naive(bundle["test_loader"], poly_idx, scaler).items():
        print(fmt_naive(name, m))
        rows.append({
            "model": name, "params_M": 0.0, "train_s": 0.0,
            "MAE_avg": m["mae_avg"], "RMSE_avg": m["rmse_avg"],
            **{f"MAE_{a}": v for a, v in zip(ASSETS, m["mae"])},
            **{f"RMSE_{a}": v for a, v in zip(ASSETS, m["rmse"])},
            "best_val_mse": np.nan,
        })

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"\n=== Training {len(model_names)} learned models ===")
    for name in model_names:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        try:
            model = build_model(name, n_ch, args.seq_len, args.pred_len).to(device)
        except Exception as e:
            print(f"  [{name}] build failed: {e}")
            traceback.print_exc()
            continue

        n_params = sum(p.numel() for p in model.parameters())
        lr = 8e-4 if name.lower() in ("prism",) else args.lr
        t0 = time.time()
        try:
            model, best_val = train_model(
                model, bundle["train_loader"], bundle["valid_loader"], device,
                epochs=args.epochs, lr=lr, patience=args.patience,
                poly_idx=poly_idx,
                max_train_batches=args.max_train_batches,
            )
            dt = time.time() - t0
            res = evaluate_polymarket(model, bundle["test_loader"], device,
                                       poly_idx, scaler, ASSETS)
            print(f"  [{name:<14}] params={n_params/1e6:.2f}M  best_val={best_val:.4f}  "
                  f"train={dt:.1f}s  MAE={res['mae_avg']:.4f}  RMSE={res['rmse_avg']:.4f}")
            rows.append({
                "model": name, "params_M": n_params / 1e6, "train_s": dt,
                "MAE_avg": res["mae_avg"], "RMSE_avg": res["rmse_avg"],
                **{f"MAE_{a}": v for a, v in zip(ASSETS, res["mae"])},
                **{f"RMSE_{a}": v for a, v in zip(ASSETS, res["rmse"])},
                "best_val_mse": best_val,
            })
        except Exception as e:
            print(f"  [{name}] training failed: {e}")
            traceback.print_exc()
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("MAE_avg")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("\n=== Final ranking (test MAE on Polymarket UP-prob) ===")
    print(f"  {'rank':>4}  {'model':<14} {'MAE':>7}  {'RMSE':>7}  {'params(M)':>10}  {'train(s)':>9}")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"  {i:>4}  {r['model']:<14} {r['MAE_avg']:>7.4f}  {r['RMSE_avg']:>7.4f}  "
              f"{r['params_M']:>10.2f}  {r['train_s']:>9.1f}")

    print(f"\nResults saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
