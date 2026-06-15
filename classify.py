"""
Classification-head experiment requested by a reviewer.

The main benchmark trains every model with MSE on the UP-token *price* and then
reads the predicted price as P(UP). MSE minimises the Brier score directly but
only indirectly the ranking-based ROC-AUC that is our headline. The reviewer
asked for at least a partial result with a dedicated *classification head*
trained under cross-entropy on the binary UP label, so discrimination is
optimised explicitly rather than inherited through the regression target.

We wrap a backbone (PRISM by default) with a small linear head that maps its
16-channel point output to one logit per asset, and train it with binary
cross-entropy against the realized UP label (sign of the realized log-return,
exactly what the contract resolves on). We then report the same proper scoring
rules (Brier, log-loss, ROC-AUC) as outcome_metrics.py, so the CE-trained head
is directly comparable to the MSE-trained model.

Example
-------
    python classify.py --backbones PRISM,Linear,LSTM --seeds 2024,2025,2026 --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_loader import create_polymarket_loaders, ASSETS
from model_zoo import build_model
from outcome_metrics import auc_score, set_seed, metrics_from

ROOT = Path(__file__).resolve().parent


class HeadClassifier(nn.Module):
    """Backbone -> 16-channel point output -> Linear(16 -> n_assets) logits."""

    def __init__(self, backbone, n_channels, n_assets):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(n_channels, n_assets)

    def forward(self, x):
        y = self.backbone(x)               # [B, pred_len, N] or [B, N]
        if y.dim() == 3:
            y = y[:, 0, :]
        return self.head(y)                # [B, n_assets] logits


def labels_from_batch(y, logret_idx, lr_mu, lr_sd):
    if y.dim() == 3:
        y = y[:, 0, :]
    realized = y[:, logret_idx].cpu().numpy() * lr_sd + lr_mu
    return (realized > 0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="2024,2025,2026")
    p.add_argument("--backbones", default="PRISM,Linear,LSTM")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", default=str(ROOT / "runs" / "classify_metrics.csv"))
    return p.parse_args()


def train_classifier(model, train_loader, valid_loader, device, epochs, lr,
                     patience, logret_idx, lr_mu, lr_sd):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            lab = torch.from_numpy(labels_from_batch(y, logret_idx, lr_mu, lr_sd)).to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for x, y in valid_loader:
                x = x.to(device)
                lab = torch.from_numpy(labels_from_batch(y, logret_idx, lr_mu, lr_sd)).to(device)
                vs += float(loss_fn(model(x), lab).item()); vn += 1
            vloss = vs / max(vn, 1)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val


@torch.no_grad()
def collect_probs(model, loader, device, logret_idx, lr_mu, lr_sd):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        probs.append(prob)
        labels.append(labels_from_batch(y, logret_idx, lr_mu, lr_sd))
    return np.concatenate(probs), np.concatenate(labels).astype(np.float64)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    device = torch.device(args.device)

    bundle = create_polymarket_loaders(
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    n_ch = bundle["n_channels"]
    poly_idx = bundle["poly_indices"]
    scaler = bundle["scaler"]
    logret_idx = [i for i, c in enumerate(scaler.columns) if c.endswith("_bin_logret")]
    lr_mu, lr_sd = scaler.mean[logret_idx], scaler.std[logret_idx]
    n_assets = len(ASSETS)
    print(f"[data] poly_idx={poly_idx} logret_idx={logret_idx} seeds={seeds}")

    lines = ["backbone,seed,brier,logloss,auc," + ",".join(f"auc_{a}" for a in ASSETS)]
    agg = {}
    for bb in backbones:
        per = []
        for seed in seeds:
            set_seed(seed)
            backbone = build_model(bb, n_ch, args.seq_len, args.pred_len)
            model = HeadClassifier(backbone, n_ch, n_assets).to(device)
            lr = 8e-4 if bb.lower() in ("prism",) else 1e-3
            model, bval = train_classifier(
                model, bundle["train_loader"], bundle["valid_loader"], device,
                args.epochs, lr, args.patience, logret_idx, lr_mu, lr_sd)
            prob, label = collect_probs(model, bundle["test_loader"], device,
                                        logret_idx, lr_mu, lr_sd)
            brier, logloss, auc = metrics_from(prob, label)
            per.append((brier.mean(), logloss.mean(), np.nanmean(auc), auc))
            lines.append(f"{bb},{seed},{brier.mean():.5f},{logloss.mean():.5f},"
                         f"{np.nanmean(auc):.5f}," + ",".join(f"{v:.5f}" for v in auc))
            print(f"  [{bb:<10}] seed={seed} BCEval={bval:.4f}  Brier={brier.mean():.4f} "
                  f"logloss={logloss.mean():.4f}  AUC={np.nanmean(auc):.4f}")
            del model, backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        b = np.array([x[0] for x in per]); l = np.array([x[1] for x in per])
        a = np.array([x[2] for x in per])
        agg[bb] = (b.mean(), b.std(), l.mean(), l.std(), a.mean(), a.std())

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Classification-head metrics (CE-trained, mean+/-std over seeds) ===")
    print(f"  {'backbone':<10} {'Brier':>16} {'logloss':>16} {'AUC':>16}")
    for name, (bm, bs, lm, ls, am, as_) in agg.items():
        print(f"  {name:<10} {bm:.4f}+/-{bs:.4f}   {lm:.4f}+/-{ls:.4f}   {am:.4f}+/-{as_:.4f}")
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
