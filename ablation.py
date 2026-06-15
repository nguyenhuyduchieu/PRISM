"""
Component ablation for PRISM on the Polymarket UP-prob task.

Each variant disables exactly one mechanism of the full PRISM model (via the
`abl_no_*` config switches added to PRISMModel) and is trained from scratch under
the *same* protocol as the main benchmark (15 epochs, AdamW + cosine, lr 8e-4,
patience 5, MSE over all 16 channels, eval MAE/RMSE on the 4 poly_up channels in
[0,1]). We run each variant over several seeds and report mean +/- std, so the
ablation is read against seed noise rather than a single lucky run.

The point is not which variant is most accurate, but to attribute PRISM's edge
to specific components: removing the cross-channel context should collapse the
model toward the channel-individual floor.

Example
-------
    python ablation.py --seeds 2024,2025,2026 --epochs 15
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
from training import train_model, evaluate_polymarket

ROOT = Path(__file__).resolve().parent

# label -> override dict for build_prism (abl_no_* switches default False = full)
VARIANTS = {
    "PRISM (full)":         {},
    "-cross-channel ctx":   {"abl_no_graph": True},
    "-hypernetwork":        {"abl_no_hyper": True},
    "-RevIN":               {"abl_no_revin": True},
    "-frequency bank":      {"abl_no_freq": True},
    "-regime":              {"abl_no_regime": True},
    "-cross-modal":         {"abl_no_crossmodal": True},
    # fully sever the cross-channel path: both the graph context (g_ctx) and the
    # channel-mixing regime latent (z), leaving each channel's forecast to depend
    # only on its own filtered history -> channel-individual PRISM
    "-all cross-channel":   {"abl_no_graph": True, "abl_no_regime": True},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", default=str(ROOT / "runs" / "ablation.csv"))
    p.add_argument("--max-train-batches", type=int, default=0,
                   help=">0 to limit train batches per epoch (smoke test)")
    p.add_argument("--only", default="",
                   help="comma-separated substrings; run only matching variants")
    return p.parse_args()


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device)
    print(f"[device] {device}  seeds={seeds}  epochs={args.epochs}")

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size,
    )
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    print(f"[data]   {len(bundle['df'])} rows, {n_ch} channels, poly idx={poly_idx}")

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    variants = {k: v for k, v in VARIANTS.items()
                if not only or any(s in k for s in only)}

    rows = []
    for label, overrides in variants.items():
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            ov = dict(overrides)
            if ov.get("abl_no_crossmodal"):
                ov["poly_idx"] = poly_idx
            model = build_prism(n_ch, args.seq_len, args.pred_len, **ov).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            t0 = time.time()
            model, best_val = train_model(
                model, bundle["train_loader"], bundle["valid_loader"], device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                poly_idx=poly_idx, max_train_batches=args.max_train_batches,
            )
            dt = time.time() - t0
            res = evaluate_polymarket(model, bundle["test_loader"], device,
                                      poly_idx, scaler, ASSETS)
            per_seed.append({
                "mae": res["mae_avg"], "rmse": res["rmse_avg"],
                **{f"mae_{a}": v for a, v in zip(ASSETS, res["mae"])},
            })
            print(f"  [{label:<20}] seed={seed}  MAE={res['mae_avg']:.4f}  "
                  f"RMSE={res['rmse_avg']:.4f}  val={best_val:.4f}  {dt:.0f}s")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        maes = np.array([d["mae"] for d in per_seed])
        rmses = np.array([d["rmse"] for d in per_seed])
        row = {
            "variant": label,
            "params_M": n_params / 1e6,
            "MAE_mean": maes.mean(), "MAE_std": maes.std(),
            "RMSE_mean": rmses.mean(), "RMSE_std": rmses.std(),
        }
        for a in ASSETS:
            vals = np.array([d[f"mae_{a}"] for d in per_seed])
            row[f"MAE_{a}_mean"] = vals.mean()
        rows.append(row)
        print(f"  => {label:<20}  MAE {maes.mean():.4f}+/-{maes.std():.4f}\n")

    df = pd.DataFrame(rows)
    full_mae = df.loc[df["variant"] == "PRISM (full)", "MAE_mean"].iloc[0]
    df["dMAE_vs_full"] = df["MAE_mean"] - full_mae
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("=== Ablation summary (MAE on Polymarket UP-prob, lower=better) ===")
    print(f"  {'variant':<20} {'MAE':>16} {'dMAE':>9} {'params(M)':>10}")
    for _, r in df.iterrows():
        print(f"  {r['variant']:<20} {r['MAE_mean']:.4f}+/-{r['MAE_std']:.4f}   "
              f"{r['dMAE_vs_full']:>+8.4f} {r['params_M']:>10.3f}")
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()
