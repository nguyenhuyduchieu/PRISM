"""
Outcome-based evaluation for the Polymarket Up/Down task.

The main benchmark scores MAE/RMSE between the predicted and realized UP-token
*price*. A reviewer-requested complement is to score the predicted UP
probability against the *realized binary event*: did the asset actually close
higher over the 15-minute window? That ground-truth label is independent of the
token price and is recovered from the sign of the realized log-return
(`bin_logret > 0`), which is exactly what the market resolves on.

For each model we treat the predicted `poly_up` (inverse-standardized, clipped
to [0,1]) as P(UP) and report proper scoring rules against the realized label:
Brier score, log-loss, and ROC-AUC (per asset + average). We also dump PRISM's
reliability curve (10 bins) for a calibration plot. Mean=0.5 is the floor
(Brier 0.25, log-loss ln 2, AUC 0.5).

Example
-------
    python outcome_metrics.py --seeds 2024,2025,2026 --epochs 15
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

ROOT = Path(__file__).resolve().parent
MODELS = ["PRISM", "LSTM", "Transformer", "Linear", "RLinear"]
EPS = 1e-6


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
    p.add_argument("--out-csv", default=str(ROOT / "runs" / "outcome_metrics.csv"))
    p.add_argument("--calib-json", default=str(ROOT / "runs" / "calibration_prism.json"))
    return p.parse_args()


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (handles ties by mid-rank)."""
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based mid-rank
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def collect(model, loader, device, poly_idx, logret_idx, scaler):
    """Return (prob_up [M,A], label [M,A]) over the whole loader."""
    model.eval()
    poly_mu, poly_sd = scaler.mean[poly_idx], scaler.std[poly_idx]
    lr_mu, lr_sd = scaler.mean[logret_idx], scaler.std[logret_idx]
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        if pred.dim() == 3:
            pred = pred[:, 0, :]
            y0 = y[:, 0, :]
        else:
            y0 = y
        p = pred[:, poly_idx].cpu().numpy() * poly_sd + poly_mu
        p = np.clip(p, 0.0, 1.0)
        realized_logret = y0[:, logret_idx].numpy() * lr_sd + lr_mu
        probs.append(p)
        labels.append((realized_logret > 0).astype(np.float64))
    return np.concatenate(probs), np.concatenate(labels)


def metrics_from(prob, label):
    """Brier / log-loss / AUC, per-asset arrays."""
    p = np.clip(prob, EPS, 1 - EPS)
    brier = ((prob - label) ** 2).mean(axis=0)
    logloss = -(label * np.log(p) + (1 - label) * np.log(1 - p)).mean(axis=0)
    auc = np.array([auc_score(label[:, a], prob[:, a]) for a in range(label.shape[1])])
    return brier, logloss, auc


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
    print(f"[data] poly_idx={poly_idx}  logret_idx={logret_idx}  seeds={seeds}")

    # base rate of UP on the test split (sanity: should be ~0.5)
    _, base_label = collect(_Mean05(n_ch, poly_idx), bundle["test_loader"], device,
                            poly_idx, logret_idx, scaler)
    print(f"[base] test UP rate per asset = {base_label.mean(axis=0).round(4)}")

    lines = ["model,seed,brier,logloss,auc," +
             ",".join(f"auc_{a}" for a in ASSETS)]
    agg = {}
    for name in models:
        per = []
        for seed in seeds:
            set_seed(seed)
            model = build_model(name, n_ch, args.seq_len, args.pred_len).to(device)
            lr = 8e-4 if name.lower() in ("prism",) else 1e-3
            model, _ = train_model(model, bundle["train_loader"], bundle["valid_loader"],
                                   device, epochs=args.epochs, lr=lr,
                                   patience=args.patience, poly_idx=poly_idx)
            prob, label = collect(model, bundle["test_loader"], device,
                                  poly_idx, logret_idx, scaler)
            brier, logloss, auc = metrics_from(prob, label)
            per.append((brier.mean(), logloss.mean(), np.nanmean(auc), auc))
            lines.append(f"{name},{seed},{brier.mean():.5f},{logloss.mean():.5f},"
                         f"{np.nanmean(auc):.5f}," + ",".join(f"{v:.5f}" for v in auc))
            print(f"  [{name:<12}] seed={seed}  Brier={brier.mean():.4f}  "
                  f"logloss={logloss.mean():.4f}  AUC={np.nanmean(auc):.4f}")
            if name.lower() in ("prism",) and seed == seeds[-1]:
                _dump_calibration(prob, label, args.calib_json)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        b = np.array([x[0] for x in per]); l = np.array([x[1] for x in per])
        a = np.array([x[2] for x in per])
        agg[name] = (b.mean(), b.std(), l.mean(), l.std(), a.mean(), a.std())

    # Mean=0.5 reference
    prob5 = np.full_like(base_label, 0.5)
    b5, l5, a5 = metrics_from(prob5, base_label)
    agg["Mean=0.5"] = (b5.mean(), 0, l5.mean(), 0, np.nanmean(a5), 0)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Outcome metrics (mean+/-std over seeds; UP=realized return>0) ===")
    print(f"  {'model':<12} {'Brier':>16} {'logloss':>16} {'AUC':>16}")
    for name, (bm, bs, lm, ls, am, as_) in agg.items():
        print(f"  {name:<12} {bm:.4f}+/-{bs:.4f}   {lm:.4f}+/-{ls:.4f}   "
              f"{am:.4f}+/-{as_:.4f}")
    print(f"\nSaved: {args.out_csv}\nCalibration: {args.calib_json}")


def _dump_calibration(prob, label, path, nbins=10):
    p = prob.ravel(); y = label.ravel()
    edges = np.linspace(0, 1, nbins + 1)
    out = []
    for k in range(nbins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < nbins - 1 else p <= edges[k + 1])
        if m.sum() > 0:
            out.append({"bin_mid": float((edges[k] + edges[k + 1]) / 2),
                        "pred_mean": float(p[m].mean()),
                        "obs_freq": float(y[m].mean()),
                        "count": int(m.sum())})
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


class _Mean05(torch.nn.Module):
    """Dummy model that returns 0.5 (standardized) for collecting labels only."""
    def __init__(self, n_ch, poly_idx):
        super().__init__()
        self.n_ch = n_ch

    def forward(self, x):
        return torch.zeros(x.size(0), 1, self.n_ch, device=x.device)


if __name__ == "__main__":
    main()
