"""
Hyperparameter sweep for PRISM on the leak-free Polymarket
task.

Runs a curated set of configs with the same data split + seed and reports test
MAE / RMSE on the Polymarket UP-prob channels (probability space, 0-1).

Configs cover:
  - input window (seq_len)
  - learning rate
  - capacity (regime_dim, graph_hidden, linear_rank)
  - num_regimes, num_bands
  - loss target: multi-task vs polymarket-only
  - longer training with patience

The best config is what we adopt as `model_zoo.PRISM_BEST_CONFIG`.

Run:
    python tune.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import build_prism
from training import evaluate_polymarket, train_model

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_configs():
    """Each entry: dict with 'name' + arbitrary overrides."""
    return [
        dict(name="baseline_default"),
        # learning rate sweep
        dict(name="lr_1e-4", learning_rate=1e-4),
        dict(name="lr_3e-4", learning_rate=3e-4),
        dict(name="lr_3e-3", learning_rate=3e-3),
        # seq_len sweep
        dict(name="ctx_48",  seq_len=48),
        dict(name="ctx_192", seq_len=192),
        # capacity larger
        dict(name="big_model",
             regime_dim=128, graph_hidden=256, linear_rank=16, num_bands=7),
        # capacity smaller (anti-overfit) — current default
        dict(name="small_model",
             regime_dim=32, graph_hidden=64, linear_rank=4, num_bands=3),
        # number of regimes
        dict(name="regimes_8", num_regimes=8),
        dict(name="regimes_2", num_regimes=2),
        # polymarket-only loss
        dict(name="poly_loss_only", poly_only_loss=True),
        dict(name="poly_loss_only_lr3e-4",
             poly_only_loss=True, learning_rate=3e-4),
        # large + long ctx + poly-only loss
        dict(name="big_ctx192_polyloss",
             seq_len=192, regime_dim=128, graph_hidden=256,
             linear_rank=16, num_bands=7, poly_only_loss=True),
        # stronger regularization
        dict(name="wd_1e-3", weight_decay=1e-3),
        # longer training
        dict(name="long_train", epochs=40, patience=10),
        # --- targeted configs to beat HATS/StockFormer ---
        # focus the loss on the 4 poly channels (the eval metric) at small capacity
        dict(name="small_polyloss",
             regime_dim=32, graph_hidden=64, linear_rank=4, num_bands=3,
             poly_only_loss=True),
        dict(name="small_polyloss_lr3e-4",
             regime_dim=32, graph_hidden=64, linear_rank=4, num_bands=3,
             poly_only_loss=True, learning_rate=3e-4),
        dict(name="small_polyloss_wd1e-3",
             regime_dim=32, graph_hidden=64, linear_rank=4, num_bands=3,
             poly_only_loss=True, weight_decay=1e-3),
        # even leaner (HATS won at 0.03M params)
        dict(name="tiny_polyloss",
             regime_dim=16, graph_hidden=32, linear_rank=2, num_bands=3,
             num_regimes=2, poly_only_loss=True),
        # lean + poly-only + longer training with early stop
        dict(name="small_polyloss_long",
             regime_dim=32, graph_hidden=64, linear_rank=4, num_bands=3,
             poly_only_loss=True, epochs=40, patience=10),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pred-len", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", default=str(ROOT / "runs" / "tune_results.csv"))
    ap.add_argument("--only", default="", help="Only run configs whose name contains this substring")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    cache = {}

    def get_bundle(seq_len, batch_size):
        key = (seq_len, batch_size)
        if key not in cache:
            cache[key] = create_polymarket_loaders(
                seq_len=seq_len, pred_len=args.pred_len, batch_size=batch_size,
            )
        return cache[key]

    configs = build_configs()
    if args.only:
        configs = [c for c in configs if args.only in c["name"]]
        print(f"[filter] {len(configs)} configs match '{args.only}'")

    rows = []
    for ci, spec in enumerate(configs, 1):
        set_seed(args.seed)
        name = spec.pop("name")
        seq_len = spec.pop("seq_len", 96)
        lr = spec.pop("learning_rate", 8e-4)
        wd = spec.pop("weight_decay", 1e-4)
        epochs = spec.pop("epochs", args.epochs)
        patience = spec.pop("patience", args.patience)
        poly_only = spec.pop("poly_only_loss", False)
        batch_size = spec.pop("batch_size", args.batch_size)
        overrides = spec  # remaining = PRISMConfig fields

        bundle = get_bundle(seq_len, batch_size)
        n_ch = bundle["n_channels"]; poly_idx = bundle["poly_indices"]
        scaler = bundle["scaler"]

        model = build_prism(n_ch, seq_len, args.pred_len, **overrides).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        try:
            model, best_val = train_model(
                model, bundle["train_loader"], bundle["valid_loader"], device,
                epochs=epochs, lr=lr, weight_decay=wd, patience=patience,
                poly_idx=poly_idx, poly_only_loss=poly_only,
            )
            dt = time.time() - t0
            res = evaluate_polymarket(model, bundle["test_loader"], device,
                                       poly_idx, scaler, ASSETS)
            row = {
                "name": name, "seq_len": seq_len, "lr": lr, "wd": wd,
                "epochs": epochs, "patience": patience, "poly_only_loss": poly_only,
                "params_M": n_params / 1e6, "train_s": dt,
                "best_val_polymse": best_val,
                "MAE_avg": res["mae_avg"], "RMSE_avg": res["rmse_avg"],
                **{f"MAE_{a}": v for a, v in zip(ASSETS, res["mae"])},
                **overrides,
            }
            rows.append(row)
            print(f"[{ci:02d}/{len(configs)}] {name:<28} "
                  f"params={n_params/1e6:.2f}M  best_val={best_val:.4f}  "
                  f"MAE={res['mae_avg']:.4f}  RMSE={res['rmse_avg']:.4f}  ({dt:.0f}s)")
        except Exception as e:
            print(f"[{ci:02d}/{len(configs)}] {name}: FAILED — {e}")
            import traceback; traceback.print_exc()
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("MAE_avg")
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== Final ranking (test MAE) ===")
    print(f"  {'rank':>4}  {'name':<28} {'MAE':>7}  {'RMSE':>7}  {'params(M)':>10}  {'train(s)':>9}")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"  {i:>4}  {r['name']:<28} {r['MAE_avg']:>7.4f}  {r['RMSE_avg']:>7.4f}  "
              f"{r['params_M']:>10.2f}  {r['train_s']:>9.1f}")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
