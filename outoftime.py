"""
Out-of-time (temporal-stability) validation of the discrimination signal.

A reviewer asked whether PRISM's small ROC-AUC edge is a property of one stretch
of the 2026 sample or holds across the test period. The whole dataset is the
2026 window split chronologically (70/15/15), so we cannot test generalization to
a different year; what we *can* test honestly is temporal stability: we cut the
held-out test period into K consecutive sub-windows and ask, per sub-window,
(i) does each model stay above the 0.5 chance level, and (ii) does PRISM's gap to
the plain Linear / LSTM baselines stay near zero, as the pooled analysis found.

We reuse the exact training/collection protocol of delong_test.py: train each
model on all three seeds, seed-average the per-instance UP probabilities, then
block over consecutive timestamps. The test_loader is unshuffled, so row order in
the [M, A] prediction matrix is chronological; row i's target lands at test
timestamp index i+seq_len, which we map back to a datetime for reporting.

Example
-------
    python outoftime.py --seeds 2024,2025,2026 --epochs 15 --blocks 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import create_polymarket_loaders, ASSETS, build_aligned_dataframe, _time_splits
from model_zoo import build_model
from training import train_model
from outcome_metrics import collect, set_seed, auc_score
from delong_test import block_bootstrap_auc, block_bootstrap_auc_diff

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--ref", default="PRISM")
    p.add_argument("--others", default="Linear,LSTM,Transformer,RLinear")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=str(ROOT / "runs" / "outoftime.json"))
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    others = [m.strip() for m in args.others.split(",") if m.strip()]
    device = torch.device(args.device)

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    logret_idx = [i for i, c in enumerate(scaler.columns) if c.endswith("_bin_logret")]

    # Recover the test-split timestamps so each prediction row gets a datetime.
    df = bundle["df"]
    _, _, test_df = _time_splits(df, (0.7, 0.15, 0.15))
    target_times = test_df.index[args.seq_len:]  # row i -> target at i+seq_len

    all_names = [args.ref] + others
    print(f"[data] poly_idx={poly_idx} logret_idx={logret_idx} seeds={seeds} blocks={args.blocks}")

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

    probs2d = {n: [] for n in all_names}
    label2d = None
    for seed in seeds:
        for name in all_names:
            prob, label = train_collect(name, seed)
            probs2d[name].append(prob)
            if label2d is None:
                label2d = label
            else:
                assert np.array_equal(label2d, label), "label mismatch across runs"
        print(f"[seed {seed}] collected {all_names}")

    avg = {n: np.mean(np.stack(probs2d[n], axis=0), axis=0) for n in all_names}
    y2d = label2d.astype(int)
    M = y2d.shape[0]

    # Consecutive temporal blocks over the M timestamps.
    edges = np.linspace(0, M, args.blocks + 1, dtype=int)
    print(f"\n=== Out-of-time stability over {args.blocks} consecutive sub-windows "
          f"(M={M} timestamps, {len(ASSETS)} assets each) ===")

    out = {"n_timestamps": int(M), "assets": ASSETS, "seeds": seeds,
           "blocks": args.blocks, "ref": args.ref, "others": others, "windows": []}

    for b in range(args.blocks):
        i0, i1 = int(edges[b]), int(edges[b + 1])
        yb = y2d[i0:i1]
        yb_flat = yb.ravel()
        t0 = str(target_times[i0]) if i0 < len(target_times) else "?"
        t1 = str(target_times[min(i1 - 1, len(target_times) - 1)]) if i1 > 0 else "?"
        rec = {"block": b, "i0": i0, "i1": i1, "n": int(yb_flat.size),
               "n_pos": int(yb_flat.sum()), "t_start": t0, "t_end": t1,
               "auc": {}, "auc_ci": {}, "dauc_vs": {}}
        print(f"\n[block {b}] rows {i0}:{i1}  n={yb_flat.size} pos={int(yb_flat.sum())}"
              f"  {t0[:16]} -> {t1[:16]}")
        ref_b = avg[args.ref][i0:i1]
        for name in all_names:
            s_b = avg[name][i0:i1]
            auc_b = auc_score(yb_flat, s_b.ravel())
            lo, hi = block_bootstrap_auc(yb, s_b, n_boot=2000, seed=100 + b)
            rec["auc"][name] = float(auc_b)
            rec["auc_ci"][name] = [float(lo), float(hi)]
            print(f"   {name:<12} AUC={auc_b:.4f}  block-boot95%=[{lo:.4f},{hi:.4f}]")
        for name in ("Linear", "LSTM"):
            if name in avg:
                s_b = avg[name][i0:i1]
                d = auc_score(yb_flat, ref_b.ravel()) - auc_score(yb_flat, s_b.ravel())
                dlo, dhi = block_bootstrap_auc_diff(yb, ref_b, s_b, n_boot=2000, seed=200 + b)
                rec["dauc_vs"][name] = {"dauc": float(d), "ci": [float(dlo), float(dhi)]}
                sep = "separates" if dlo > 0 else "tied"
                print(f"   dAUC({args.ref}-{name})={d:+.4f}  "
                      f"block-boot95%=[{dlo:+.4f},{dhi:+.4f}]  [{sep}]")
        out["windows"].append(rec)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
