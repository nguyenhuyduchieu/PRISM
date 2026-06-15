"""
Naive baselines for the Polymarket UP-probability forecasting task.

  - Persistence : pred = last observed Polymarket value (broadcast across horizon)
  - Mean = 0.5  : uniform-prior probability

These set the sanity floor for the learned models.

Example
-------
    python naive_baselines.py --seq-len 96 --pred-len 1
"""

from __future__ import annotations

import argparse

from data_loader import create_polymarket_loaders, ASSETS
from training import evaluate_naive


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size,
    )
    metrics = evaluate_naive(bundle["test_loader"], bundle["poly_indices"], bundle["scaler"])

    print("=== Naive baselines on Polymarket UP-prob (test split, 0-1 space) ===")
    print(f"  {'asset':<6}  {'persist MAE':>12}  {'persist RMSE':>13}  "
          f"{'mean MAE':>10}  {'mean RMSE':>11}")
    for j, a in enumerate(ASSETS):
        print(f"  {a.upper():<6}  "
              f"{metrics['Persistence']['mae'][j]:>12.4f}  "
              f"{metrics['Persistence']['rmse'][j]:>13.4f}  "
              f"{metrics['Mean=0.5']['mae'][j]:>10.4f}  "
              f"{metrics['Mean=0.5']['rmse'][j]:>11.4f}")
    print(f"  {'ALL':<6}  "
          f"{metrics['Persistence']['mae_avg']:>12.4f}  "
          f"{metrics['Persistence']['rmse_avg']:>13.4f}  "
          f"{metrics['Mean=0.5']['mae_avg']:>10.4f}  "
          f"{metrics['Mean=0.5']['rmse_avg']:>11.4f}")


if __name__ == "__main__":
    main()
