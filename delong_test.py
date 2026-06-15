"""
DeLong significance test for the outcome-metric (ROC-AUC) comparison.

The benchmark reports ROC-AUC mean+/-std over seeds, but a reviewer asked for a
proper significance test on the AUC gap between PRISM and the next-best model.
We add the standard DeLong test for two *correlated* ROC curves (DeLong, DeLong &
Clarke-Pearson 1988; fast midrank algorithm of Sun & Xu 2014), which is exact for
paired predictions on the same test set.

For each seed we train PRISM and the comparison models under the identical
protocol used in outcome_metrics.py, pool the four assets' UP-probability
predictions against the realized binary UP label, and compute:
  * each model's pooled AUC,
  * the two-sided DeLong p-value for AUC(PRISM) - AUC(other),
  * a paired bootstrap 95% CI for the same AUC difference (sanity cross-check).

Example
-------
    python delong_test.py --seeds 2024,2025,2026 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import build_model
from training import train_model
from outcome_metrics import collect, set_seed, auc_score

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# DeLong machinery (fast midrank algorithm)
# ---------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted: np.ndarray, n_pos: int):
    """preds_sorted: [2, N] scores, positives in the first n_pos columns."""
    m = n_pos
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """Two-sided DeLong p-value for AUC(a) - AUC(b) on paired predictions."""
    y = y_true.astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives (label 1) first
    n_pos = int(y.sum())
    preds = np.vstack((score_a, score_b))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T)[0, 0])
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        z = 0.0 if diff == 0 else math.copysign(float("inf"), diff)
        p = 0.0 if math.isinf(z) else 1.0
    else:
        z = diff / math.sqrt(var)
        p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided normal
    return float(aucs[0]), float(aucs[1]), diff, z, p


def bootstrap_auc_diff(y, sa, sb, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            diffs[i] = np.nan
            continue
        diffs[i] = auc_score(yb, sa[idx]) - auc_score(yb, sb[idx])
    diffs = diffs[~np.isnan(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------------
# Permutation + block-bootstrap on pooled (seed-averaged) predictions
# ---------------------------------------------------------------------------
def permutation_auc_diff(y, sa, sb, n_perm=10000, seed=0):
    """Two-sided permutation p for AUC(a)-AUC(b) under exchangeability of the
    paired scores: for each observation, randomly swap (sa, sb)."""
    rng = np.random.default_rng(seed)
    obs = auc_score(y, sa) - auc_score(y, sb)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap, sb, sa)
        b = np.where(swap, sa, sb)
        d = auc_score(y, a) - auc_score(y, b)
        if abs(d) >= abs(obs) - 1e-12:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)


def permutation_auc_gt_half(y, s, n_perm=10000, seed=0):
    """One-sided permutation p for H0: AUC=0.5 vs H1: AUC>0.5, by shuffling the
    binary label (equivalent to the Mann-Whitney null)."""
    rng = np.random.default_rng(seed)
    obs = auc_score(y, s)
    yb = y.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(yb)
        if auc_score(yb, s) >= obs - 1e-12:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)


def block_bootstrap_auc(y2d, s2d, n_boot=2000, seed=0):
    """Block bootstrap CI for a single pooled AUC, resampling whole timestamps
    (rows of [M, A]) so cross-asset dependence within a timestamp is preserved."""
    rng = np.random.default_rng(seed)
    M = y2d.shape[0]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, M, M)
        yb = y2d[rows].ravel()
        sb = s2d[rows].ravel()
        aucs[i] = auc_score(yb, sb) if 0 < yb.sum() < len(yb) else np.nan
    aucs = aucs[~np.isnan(aucs)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def block_bootstrap_auc_diff(y2d, a2d, b2d, n_boot=2000, seed=0):
    """Block bootstrap CI for AUC(a)-AUC(b), resampling whole timestamps."""
    rng = np.random.default_rng(seed)
    M = y2d.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, M, M)
        yb = y2d[rows].ravel()
        if not (0 < yb.sum() < len(yb)):
            diffs[i] = np.nan
            continue
        diffs[i] = auc_score(yb, a2d[rows].ravel()) - auc_score(yb, b2d[rows].ravel())
    diffs = diffs[~np.isnan(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=str(ROOT / "runs" / "delong_test.json"))
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
    print(f"[data] poly_idx={poly_idx}  logret_idx={logret_idx}  seeds={seeds}")

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
        return prob, label  # 2D [M, A]

    all_names = [args.ref] + others
    probs2d = {n: [] for n in all_names}  # name -> list of [M,A] over seeds
    label2d_ref = None

    results = []
    for seed in seeds:
        ref_prob, ref_label = train_collect(args.ref, seed)
        probs2d[args.ref].append(ref_prob)
        if label2d_ref is None:
            label2d_ref = ref_label
        ref_flat, lab_flat = ref_prob.ravel(), ref_label.ravel()
        auc_ref = auc_score(lab_flat.astype(int), ref_flat)
        print(f"\n[seed {seed}] {args.ref} pooled AUC = {auc_ref:.4f}  "
              f"(n={len(lab_flat)}, pos={int(lab_flat.sum())})")
        for name in others:
            o_prob, o_label = train_collect(name, seed)
            probs2d[name].append(o_prob)
            assert np.array_equal(ref_label, o_label), "label mismatch across models"
            o_flat = o_prob.ravel()
            a_ref, a_oth, diff, z, p = delong_test(lab_flat, ref_flat, o_flat)
            lo, hi = bootstrap_auc_diff(lab_flat.astype(int), ref_flat, o_flat)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {args.ref} vs {name:<12} "
                  f"AUC {a_ref:.4f} vs {a_oth:.4f}  dAUC={diff:+.4f}  "
                  f"z={z:+.3f}  p={p:.4f} [{sig}]  boot95%=[{lo:+.4f},{hi:+.4f}]")
            results.append({"seed": seed, "ref": args.ref, "other": name,
                            "auc_ref": a_ref, "auc_other": a_oth, "dauc": diff,
                            "z": z, "p_value": p, "boot_lo": lo, "boot_hi": hi})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved per-seed: {args.out_json}")

    # ---------------------------------------------------------------------
    # Pooled test on seed-averaged predictions (one independent test set,
    # labels appear once). 2D arrays keep the [timestamp, asset] structure
    # so the block bootstrap can resample whole timestamps.
    # ---------------------------------------------------------------------
    print("\n=== Pooled (seed-averaged) significance on the headline AUC ===")
    y2d = label2d_ref.astype(int)
    y_flat = y2d.ravel()
    avg = {n: np.mean(np.stack(probs2d[n], axis=0), axis=0) for n in all_names}

    ref2d = avg[args.ref]
    auc_ref, p_ref = permutation_auc_gt_half(y_flat, ref2d.ravel(), seed=0)
    lo_ref, hi_ref = block_bootstrap_auc(y2d, ref2d, seed=1)
    print(f"{args.ref} pooled AUC = {auc_ref:.4f}  "
          f"block-boot95%=[{lo_ref:.4f},{hi_ref:.4f}]  "
          f"perm p(AUC>0.5)={p_ref:.4g}  (n={len(y_flat)}, pos={int(y_flat.sum())})")

    pooled = {"n": int(len(y_flat)), "n_pos": int(y_flat.sum()),
              "seeds": seeds, "ref": args.ref,
              "auc_ref": auc_ref, "ref_boot_lo": lo_ref, "ref_boot_hi": hi_ref,
              "ref_perm_p_gt_half": p_ref, "comparisons": []}

    for name in others:
        o2d = avg[name]
        a_ref, a_oth, diff, z, p_dl = delong_test(y_flat, ref2d.ravel(), o2d.ravel())
        _, p_perm = permutation_auc_diff(y_flat, ref2d.ravel(), o2d.ravel(), seed=2)
        blo, bhi = block_bootstrap_auc_diff(y2d, ref2d, o2d, seed=3)
        auc_o, p_o = permutation_auc_gt_half(y_flat, o2d.ravel(), seed=4)
        sig = "***" if p_dl < 0.001 else "**" if p_dl < 0.01 else "*" if p_dl < 0.05 else "ns"
        print(f"  {args.ref} vs {name:<12} AUC {a_ref:.4f} vs {a_oth:.4f}  "
              f"dAUC={diff:+.4f}  DeLong p={p_dl:.4g}[{sig}]  perm p={p_perm:.4g}  "
              f"block-boot95%=[{blo:+.4f},{bhi:+.4f}]")
        pooled["comparisons"].append(
            {"other": name, "auc_ref": a_ref, "auc_other": a_oth, "dauc": diff,
             "z": z, "delong_p": p_dl, "perm_p": p_perm,
             "boot_lo": blo, "boot_hi": bhi, "other_auc_perm_p_gt_half": p_o})

    _stem = Path(args.out_json).stem
    pooled_path = str(Path(args.out_json).with_name(f"{_stem}_pooled.json"))
    Path(pooled_path).write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    print(f"\nSaved pooled: {pooled_path}")


if __name__ == "__main__":
    main()
