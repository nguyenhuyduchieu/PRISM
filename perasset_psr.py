"""
Per-asset proper-scoring-rule table (reviewer R3, camera-ready #3a, full version).

outcome_metrics.py saves per-asset AUC but only the asset-averaged Brier/log-loss.
This script recomputes the *per-asset* Brier, log-loss, and ROC-AUC for the five
report models over three seeds, so the appendix table can show all three metrics
per asset. It reuses outcome_metrics.collect / metrics_from / set_seed verbatim,
so the numbers are identical in construction to the pooled table.

Run
---
    python perasset_psr.py --seeds 2024,2025,2026 --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import build_model
from training import train_model
from outcome_metrics import collect, metrics_from, set_seed

ROOT = Path(__file__).resolve().parent
MODELS = ["PRISM", "Linear", "LSTM", "Transformer", "RLinear"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=str(ROOT / "runs" / "perasset_psr.json"))
    p.add_argument("--out-csv", default=str(ROOT / "runs" / "perasset_psr.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device(args.device)

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    logret_idx = [i for i, c in enumerate(scaler.columns) if c.endswith("_bin_logret")]
    A = len(ASSETS)

    rows = ["model,seed,metric," + ",".join(ASSETS) + ",avg"]
    agg = {}
    for name in models:
        per = {"brier": [], "logloss": [], "auc": []}
        for seed in seeds:
            set_seed(seed)
            model = build_model(name, n_ch, args.seq_len, args.pred_len).to(device)
            lr = 8e-4 if name.lower() in ("prism",) else 1e-3
            model, _ = train_model(model, bundle["train_loader"], bundle["valid_loader"],
                                   device, epochs=args.epochs, lr=lr,
                                   patience=args.patience, poly_idx=poly_idx)
            prob, label = collect(model, bundle["test_loader"], device,
                                  poly_idx, logret_idx, scaler)
            brier, logloss, auc = metrics_from(prob, label)   # each [A]
            per["brier"].append(brier); per["logloss"].append(logloss); per["auc"].append(auc)
            for mname, arr in (("brier", brier), ("logloss", logloss), ("auc", auc)):
                rows.append(f"{name},{seed},{mname}," +
                            ",".join(f"{v:.5f}" for v in arr) + f",{np.nanmean(arr):.5f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # mean over seeds, per asset
        agg[name] = {m: np.nanmean(np.stack(per[m], axis=0), axis=0).tolist()
                     for m in per}
        print(f"[{name}] done over {len(seeds)} seeds")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).write_text("\n".join(rows), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # ---- pretty print + LaTeX-ready means ----
    print("\n=== Per-asset means over seeds ===")
    for metric, arrow in (("brier", "v"), ("logloss", "v"), ("auc", "^")):
        print(f"\n-- {metric} ({arrow}) --   " + "  ".join(f"{a:>7}" for a in ASSETS) + "      avg")
        for name in models:
            v = agg[name][metric]
            print(f"  {name:<12} " + "  ".join(f"{x:7.4f}" for x in v) +
                  f"   {float(np.mean(v)):7.4f}")
    print(f"\nSaved: {args.out_csv}\n       {args.out_json}")


if __name__ == "__main__":
    main()
