"""
Forward-fill leakage / robustness audit (reviewer R1, camera-ready required #1).

Two questions from the review:
  (a) What fraction of the aligned 15-minute buckets were forward-filled?
  (b) Does excluding test windows whose *target bucket* was forward-filled change
      the pooled ROC-AUC the paper's headline rests on?

Why this matters: on a near-coin-flip target, a forward-filled target bucket
duplicates the previous bucket's value. If the label (sign of the realized
`bin_logret`) or the predicted `poly_up` at the target is a stale copy of a value
already inside the input window, a model could score above chance trivially. We
quantify how often that happens and recompute the pooled AUC with those windows
removed.

The fill mask is reconstructed by replaying `build_aligned_dataframe`'s exact
ffill(limit=4)+dropna logic while recording which surviving cells were NaN
*before* the fill. Test windows map to target rows deterministically because the
test loader uses shuffle=False.

Run
---
    python forwardfill_audit.py --stats-only          # fast: just the fractions
    python forwardfill_audit.py --seeds 2024,2025,2026 # full: + re-pooled AUC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import (
    ASSETS, FEATURES, POLY_DIR_DEFAULT, BIN_DIR_DEFAULT,
    _load_poly_up_series, _load_binance_15m, create_polymarket_loaders,
)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Rebuild the aligned frame while tracking which cells were forward-filled.
# Mirrors data_loader.build_aligned_dataframe exactly.
# ---------------------------------------------------------------------------
def build_with_fillmask(assets=ASSETS, poly_dir=POLY_DIR_DEFAULT, bin_dir=BIN_DIR_DEFAULT):
    pieces = []
    for a in assets:
        poly = _load_poly_up_series(a, poly_dir).to_frame()
        binf = _load_binance_15m(a, bin_dir)
        merged = poly.join(binf, how="outer")
        merged = merged[[f"{a}_{feat}" for feat in FEATURES]]
        pieces.append(merged)

    aligned = pieces[0]
    for p in pieces[1:]:
        aligned = aligned.join(p, how="outer")

    poly_cols = [f"{a}_poly_up" for a in assets]
    bin_cols = [c for c in aligned.columns if c not in poly_cols]

    poly_mask = aligned[poly_cols].notna().all(axis=1)
    if poly_mask.any():
        first_valid = aligned.index[poly_mask].min()
        last_valid = aligned.index[poly_mask].max()
        aligned = aligned.loc[first_valid:last_valid]

    pre = aligned.copy()                       # NaN here == originally missing
    aligned[poly_cols] = aligned[poly_cols].ffill(limit=4)
    aligned[bin_cols] = aligned[bin_cols].ffill(limit=4)
    keep = aligned.dropna()
    # a surviving cell that was NaN before ffill must have been filled by ffill
    fillmask = pre.loc[keep.index].isna()
    return keep, fillmask, poly_cols, bin_cols


def report_fractions(keep, fillmask, poly_cols, bin_cols):
    n = len(keep)
    logret_cols = [f"{a}_bin_logret" for a in ASSETS]
    out = {"n_rows": int(n), "per_column": {}}
    print(f"\n=== Forward-fill accounting on the aligned frame ===")
    print(f"aligned rows (buckets): {n}  range {keep.index.min()} -> {keep.index.max()}")
    print(f"{'column':<20} {'filled':>8} {'pct':>8}")
    for c in keep.columns:
        f = int(fillmask[c].sum())
        out["per_column"][c] = {"filled": f, "pct": 100.0 * f / n}
        print(f"  {c:<18} {f:>8} {100.0*f/n:>7.3f}%")

    any_row = int(fillmask.any(axis=1).sum())
    poly_row = int(fillmask[poly_cols].any(axis=1).sum())
    bin_row = int(fillmask[bin_cols].any(axis=1).sum())
    logret_row = int(fillmask[logret_cols].any(axis=1).sum())
    cell_total = int(fillmask.values.sum())
    cell_frac = 100.0 * cell_total / (n * fillmask.shape[1])

    print(f"\nrows with >=1 filled cell : {any_row}  ({100.0*any_row/n:.3f}%)")
    print(f"rows with filled poly_up  : {poly_row}  ({100.0*poly_row/n:.3f}%)")
    print(f"rows with filled bin_*    : {bin_row}  ({100.0*bin_row/n:.3f}%)")
    print(f"rows with filled bin_logret: {logret_row}  ({100.0*logret_row/n:.3f}%)")
    print(f"filled cells overall      : {cell_total}/{n*fillmask.shape[1]} ({cell_frac:.4f}%)")

    out.update(dict(rows_any_filled=any_row, rows_poly_filled=poly_row,
                    rows_bin_filled=bin_row, rows_logret_filled=logret_row,
                    cells_filled=cell_total, cells_pct=cell_frac))
    return out


def window_fill_flags(keep, fillmask, seq_len=96, pred_len=1,
                      ratios=(0.7, 0.15, 0.15)):
    """For each test window, flag whether its TARGET bucket was forward-filled.

    Window m (0-based, loader order) predicts test row (m + seq_len). Returns
      excl_window[m]      : any of the 4 assets' poly_up/bin_logret target filled
      excl_inst[m, a]     : asset a's poly_up or bin_logret target filled
    """
    n = len(keep)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    test_start = n_train + n_valid
    n_test = n - test_start
    M = max(0, n_test - seq_len - pred_len + 1)

    fm = fillmask.values  # [n, 16] bool, column order == keep.columns
    cols = list(keep.columns)
    poly_pos = [cols.index(f"{a}_poly_up") for a in ASSETS]
    lr_pos = [cols.index(f"{a}_bin_logret") for a in ASSETS]

    excl_inst = np.zeros((M, len(ASSETS)), dtype=bool)
    for m in range(M):
        gidx = test_start + m + seq_len           # global positional row of target
        for ai in range(len(ASSETS)):
            excl_inst[m, ai] = fm[gidx, poly_pos[ai]] or fm[gidx, lr_pos[ai]]
    excl_window = excl_inst.any(axis=1)
    return M, excl_window, excl_inst


# ---------------------------------------------------------------------------
# AUC recomputation with/without forward-filled target windows
# ---------------------------------------------------------------------------
def auc_score(y_true, score):
    """Mann-Whitney ROC-AUC with mid-rank tie handling (same as outcome_metrics)."""
    y_true = y_true.astype(int)
    n_pos = float(y_true.sum()); n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def block_bootstrap_auc(y2d, s2d, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    M = y2d.shape[0]
    a = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, M, M)
        yb = y2d[rows].ravel(); sb = s2d[rows].ravel()
        a[i] = auc_score(yb, sb) if 0 < yb.sum() < len(yb) else np.nan
    a = a[~np.isnan(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run_auc(models, seeds, excl_window, args):
    from model_zoo import build_model
    from training import train_model
    from outcome_metrics import collect, set_seed

    device = torch.device(args.device)
    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    logret_idx = [i for i, c in enumerate(scaler.columns) if c.endswith("_bin_logret")]
    n_ch = bundle["n_channels"]

    keep_mask = ~excl_window                       # True = keep (not forward-filled)
    results = {"n_windows_total": int(len(excl_window)),
               "n_windows_excluded": int(excl_window.sum()),
               "n_windows_kept": int(keep_mask.sum()),
               "models": {}}
    print(f"\n=== Re-pooled AUC (seed-averaged) ===")
    print(f"test windows: {len(excl_window)}  excluded(ffilled target): "
          f"{int(excl_window.sum())}  kept: {int(keep_mask.sum())}")

    for name in models:
        probs = []
        label2d = None
        for seed in seeds:
            set_seed(seed)
            model = build_model(name, n_ch, args.seq_len, args.pred_len).to(device)
            lr = 8e-4 if name.lower() in ("prism",) else 1e-3
            model, _ = train_model(model, bundle["train_loader"], bundle["valid_loader"],
                                   device, epochs=args.epochs, lr=lr,
                                   patience=args.patience, poly_idx=poly_idx)
            prob, label = collect(model, bundle["test_loader"], device,
                                  poly_idx, logret_idx, scaler)
            probs.append(prob)
            label2d = label
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        prob2d = np.mean(np.stack(probs, axis=0), axis=0)   # [M, A]
        y2d = label2d.astype(int)

        # sanity: M must match the fill-flag count
        if prob2d.shape[0] != len(excl_window):
            raise SystemExit(f"window count mismatch: preds {prob2d.shape[0]} "
                             f"vs flags {len(excl_window)}")

        auc_all = auc_score(y2d.ravel(), prob2d.ravel())
        lo_all, hi_all = block_bootstrap_auc(y2d, prob2d, seed=1)
        yk, pk = y2d[keep_mask], prob2d[keep_mask]
        auc_ex = auc_score(yk.ravel(), pk.ravel())
        lo_ex, hi_ex = block_bootstrap_auc(yk, pk, seed=1)

        results["models"][name] = dict(
            auc_all=auc_all, ci_all=[lo_all, hi_all],
            auc_excl=auc_ex, ci_excl=[lo_ex, hi_ex], delta=auc_ex - auc_all)
        print(f"  {name:<10} AUC(all)={auc_all:.4f} [{lo_all:.4f},{hi_all:.4f}]   "
              f"AUC(excl-ffill)={auc_ex:.4f} [{lo_ex:.4f},{hi_ex:.4f}]   "
              f"delta={auc_ex-auc_all:+.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--models", default="PRISM,Linear,LSTM")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=str(ROOT / "runs" / "forwardfill_audit.json"))
    return p.parse_args()


def main():
    args = parse_args()
    keep, fillmask, poly_cols, bin_cols = build_with_fillmask()
    frac = report_fractions(keep, fillmask, poly_cols, bin_cols)
    M, excl_window, excl_inst = window_fill_flags(
        keep, fillmask, seq_len=args.seq_len, pred_len=args.pred_len)
    inst_excl = int(excl_inst.sum()); inst_tot = excl_inst.size
    print(f"\ntest windows (L={args.seq_len}): {M}")
    print(f"  windows with forward-filled target : {int(excl_window.sum())} "
          f"({100.0*excl_window.sum()/M:.3f}%)")
    print(f"  (window,asset) instances filled    : {inst_excl}/{inst_tot} "
          f"({100.0*inst_excl/inst_tot:.3f}%)")

    payload = {"fractions": frac,
               "test_windows": int(M),
               "windows_target_filled": int(excl_window.sum()),
               "windows_target_filled_pct": 100.0 * excl_window.sum() / M,
               "instances_filled": inst_excl,
               "instances_total": int(inst_tot),
               "instances_filled_pct": 100.0 * inst_excl / inst_tot}

    if not args.stats_only:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        payload["auc"] = run_auc(models, seeds, excl_window, args)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
