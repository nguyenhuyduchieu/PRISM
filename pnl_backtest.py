"""
Economic / PnL backtest after fees and spread.

A reviewer noted that we motivate the work with market-making value, so an
above-chance AUC is not enough---the edge must survive transaction costs. We
turn each model's predicted UP probability into a directional position on the
next 15-minute contract and book the realized PnL net of a configurable fee and
bid--ask spread.

Setup
-----
For each test instance we know:
  * p_hat  : the model's predicted P(UP) for the contract resolving at t+1,
  * y      : the realized binary UP outcome (pays 1 to the winning side).

We do NOT have the live mid-price of the contract we are betting on at entry
time: the only Polymarket price in the feature set, poly_up[t], is the *closing*
tick of the market that just resolved at t, which is essentially degenerate
(near 0 or near 1, the realized outcome) and is therefore neither a valid entry
price for the next market nor leakage-free. The honest economic test is thus
against the fair martingale price of 0.5: a directional position is profitable
only if the model's directional skill clears the transaction cost of a
0.5-priced contract.

Strategy (per asset, per timestamp), with entry price 0.5 and edge threshold tau:
  * if p_hat - 0.5 >  tau : BUY UP   at cost = 0.5 + spread/2, payoff = y
  * if p_hat - 0.5 < -tau : BUY DOWN at cost = 0.5 + spread/2, payoff = 1-y
  * else: no trade.
Per-trade PnL = payoff - cost - fee*cost (stake = 1 contract). We report total
PnL, #trades, hit rate, mean PnL/trade and its bootstrap CI, for a grid of cost
levels and thresholds. A "no-cost, vs-0.5" variant (anchor=0.5, fee=spread=0) is
included as the theoretical ceiling.

Example
-------
    python pnl_backtest.py --models PRISM,Linear,LSTM --seeds 2024,2025,2026 --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import create_polymarket_loaders, ASSETS, _time_splits
from model_zoo import build_model
from training import train_model
from outcome_metrics import collect, set_seed

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--models", default="PRISM,Linear,LSTM")
    p.add_argument("--thresholds", default="0.0,0.02,0.05")
    p.add_argument("--cost-grid", default="0/0,0.0/0.01,0.0/0.02,0.02/0.02",
                   help="semicolon-free list of fee/spread pairs, comma-separated")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=str(ROOT / "runs" / "pnl_backtest.json"))
    return p.parse_args()


def boot_mean_ci(x, n_boot=2000, seed=0):
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def backtest(p_hat, y, tau, fee, spread, entry=0.5):
    """Vectorized over flattened [N] arrays. Entry at the fair price `entry`.
    Returns per-trade PnL array + #trades + #hits."""
    edge = p_hat - entry
    buy_up = edge > tau
    buy_dn = edge < -tau
    cost = entry + spread / 2.0
    pnl_up = y[buy_up] - cost - fee * cost
    pnl_dn = (1.0 - y[buy_dn]) - cost - fee * cost
    pnl = np.concatenate([pnl_up, pnl_dn]) if (buy_up.any() or buy_dn.any()) else np.array([])
    n_tr = int(buy_up.sum() + buy_dn.sum())
    hits = int(y[buy_up].sum() + (1 - y[buy_dn]).sum()) if n_tr else 0
    return pnl, n_tr, hits


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    taus = [float(t) for t in args.thresholds.split(",") if t.strip()]
    costs = []
    for pair in args.cost_grid.split(","):
        f, s = pair.split("/")
        costs.append((float(f), float(s)))
    device = torch.device(args.device)

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    logret_idx = [i for i, c in enumerate(scaler.columns) if c.endswith("_bin_logret")]

    def train_collect(name, seed):
        set_seed(seed)
        model = build_model(name, n_ch, args.seq_len, args.pred_len).to(device)
        lr = 8e-4 if name.lower() in ("prism",) else 1e-3
        model, _ = train_model(model, bundle["train_loader"], bundle["valid_loader"],
                               device, epochs=args.epochs, lr=lr,
                               patience=args.patience, poly_idx=poly_idx)
        prob, label = collect(model, bundle["test_loader"], device,
                              poly_idx, logret_idx, scaler)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prob, label

    probs2d = {m: [] for m in models}
    label2d = None
    for seed in seeds:
        for name in models:
            prob, label = train_collect(name, seed)
            probs2d[name].append(prob)
            if label2d is None:
                label2d = label
            else:
                assert np.array_equal(label2d, label), "label mismatch"
        print(f"[seed {seed}] collected {models}")

    avg = {m: np.mean(np.stack(probs2d[m], axis=0), axis=0) for m in models}
    y2d = label2d.astype(np.float64)
    y_flat = y2d.ravel()
    n_all = len(y_flat)
    np.savez(Path(args.out_json).with_suffix(".npz"),
             y=y_flat, **{f"prob_{m}": avg[m].ravel() for m in models})

    out = {"n": int(n_all), "seeds": seeds, "models": models,
           "thresholds": taus, "cost_grid": [{"fee": f, "spread": s} for f, s in costs],
           "results": []}

    print(f"\n=== PnL backtest (n={n_all} asset-windows, base UP rate={y_flat.mean():.3f}) ===")
    for name in models:
        p_flat = avg[name].ravel()
        print(f"\n--- {name} ---")
        for (fee, spread) in costs:
            for tau in taus:
                pnl, n_tr, hits = backtest(p_flat, y_flat, tau, fee, spread)
                if n_tr == 0:
                    print(f"  fee={fee} spread={spread} tau={tau}: no trades")
                    continue
                total = float(pnl.sum())
                mean = float(pnl.mean())
                lo, hi = boot_mean_ci(pnl, seed=7)
                hr = hits / n_tr
                roi = total / n_tr  # per-trade ROI (stake ~ cost ~ O(0.5))
                profitable = lo > 0
                print(f"  fee={fee:<4} spread={spread:<4} tau={tau:<4} "
                      f"trades={n_tr:<5} hit={hr:.3f} totalPnL={total:+.2f} "
                      f"PnL/trade={mean:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                      f"{'PROFITABLE' if profitable else ''}")
                out["results"].append(
                    {"model": name, "fee": fee, "spread": spread, "tau": tau,
                     "trades": n_tr, "hit_rate": hr, "total_pnl": total,
                     "pnl_per_trade": mean, "pnl_ci": [lo, hi],
                     "profitable_at_95": bool(profitable)})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
