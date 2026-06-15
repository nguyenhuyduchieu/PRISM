"""
Reproducible audit of temporal alignment between Polymarket and Binance.

What it checks
--------------
The 15-minute Polymarket "Up/Down" market at time `t` is the market that
RESOLVES at `t`, i.e. it was active over `(t-15min, t]`. Its UP-token price
`poly_up[t]` is therefore observable AT `t`.

The Binance features come from a 15m resample of 1m OHLCV. Pandas's default for
`resample("15min")` is `closed='left', label='left'`, which means the bin
labeled `t` covers `[t, t+15min)` — i.e. FUTURE information w.r.t. time `t`.

If we feed `[poly_up[t], bin_*[t], ...]` as the model's input row at time `t`
and ask it to predict `poly_up[t+1]`, the model effectively gets to look at
Binance returns DURING the next Polymarket market — a textbook lookahead leak.

Signature: `corr(bin_logret[t], poly_up[t+1])` is large and positive (≈ +0.63)
under the broken alignment. After fixing the resample to
`closed='right', label='right'` (so the bin labeled `t` covers `(t-15min, t]`),
the same correlation collapses to ≈ -0.01.

Run
---
    python audit_leakage.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data_loader import (
    ASSETS, BIN_DIR_DEFAULT, POLY_DIR_DEFAULT, _load_poly_up_series,
    build_aligned_dataframe,
)


def _load_binance_buggy(asset: str, bin_dir: str) -> pd.DataFrame:
    """Reproduce the BROKEN alignment (resample left-closed, left-labeled)."""
    path = os.path.join(bin_dir, f"{asset.upper()}USDT.csv")
    df = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    # default: closed='left', label='left'
    agg = df.resample("15min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    close = agg["Close"].astype("float32")
    out = pd.DataFrame(index=agg.index)
    out[f"{asset}_bin_logret"] = np.log(close / close.shift(1)).astype("float32")
    out[f"{asset}_bin_logvol"] = np.log1p(agg["Volume"].astype("float32"))
    rng = (agg["High"] - agg["Low"]) / agg["Close"].replace(0, np.nan)
    out[f"{asset}_bin_range"] = rng.astype("float32")
    return out.dropna()


def build_buggy_aligned_dataframe(assets=ASSETS,
                                   poly_dir=POLY_DIR_DEFAULT,
                                   bin_dir=BIN_DIR_DEFAULT) -> pd.DataFrame:
    """Re-build the aligned dataframe with the buggy Binance alignment."""
    from data_loader import FEATURES
    pieces = []
    for a in assets:
        poly = _load_poly_up_series(a, poly_dir).to_frame()
        binf = _load_binance_buggy(a, bin_dir)
        merged = poly.join(binf, how="outer")
        merged = merged[[f"{a}_{f}" for f in FEATURES]]
        pieces.append(merged)
    aligned = pieces[0]
    for p in pieces[1:]:
        aligned = aligned.join(p, how="outer")
    poly_cols = [f"{a}_poly_up" for a in assets]
    poly_mask = aligned[poly_cols].notna().all(axis=1)
    if poly_mask.any():
        aligned = aligned.loc[aligned.index[poly_mask].min() : aligned.index[poly_mask].max()]
    aligned = aligned.ffill(limit=4).dropna()
    return aligned


def report_leak(df: pd.DataFrame, label: str):
    print(f"\n--- {label} ---")
    print(f"shape: {df.shape}  range: {df.index.min()} .. {df.index.max()}")
    rows = []
    for a in ASSETS:
        nxt = df[f"{a}_poly_up"].shift(-1)
        for feat in ("bin_logret", "bin_logvol", "bin_range"):
            c = df[f"{a}_{feat}"].corr(nxt)
            rows.append((a.upper(), feat, c))
    print(f"  {'asset':<5} {'feature':<12} {'corr(feat[t], poly_up[t+1])':>30}")
    for a, f, c in rows:
        flag = "  <-- LEAK" if abs(c) > 0.1 else ""
        print(f"  {a:<5} {f:<12} {c:>+30.4f}{flag}")
    # aggregate logret stat
    logret_cors = [c for (_, f, c) in rows if f == "bin_logret"]
    print(f"  mean |corr(bin_logret[t], poly_up[t+1])| across assets = "
          f"{np.mean(np.abs(logret_cors)):.4f}")


def main():
    print("=" * 72)
    print("Polymarket / Binance temporal-alignment leakage audit")
    print("=" * 72)

    buggy = build_buggy_aligned_dataframe()
    report_leak(
        buggy,
        "BUGGY: resample default (closed='left', label='left')\n"
        "  bin labeled t covers [t, t+15min) -> FUTURE info vs market ending at t",
    )

    fixed = build_aligned_dataframe()
    report_leak(
        fixed,
        "FIXED: resample(closed='right', label='right')\n"
        "  bin labeled t covers (t-15min, t] -> OBSERVABLE at t",
    )

    print("\nExpected:")
    print("  - BUGGY mean |corr| ~ 0.6 (clear lookahead)")
    print("  - FIXED mean |corr| < 0.05 (noise)")
    print("\nSee docs/leakage_audit.md for a full writeup.")


if __name__ == "__main__":
    main()
