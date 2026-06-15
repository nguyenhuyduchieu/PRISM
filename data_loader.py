"""
Aligned multimodal data loader for PRISM (and 10 baselines) on Polymarket
"Up/Down 15m" markets.

Inputs
------
- Polymarket: per-asset parquet (tick-level YES/NO token prices, 15m markets).
  Source: $POLY_DIR/{asset}_prices.parquet , $POLY_DIR/{asset}_metadata.parquet
- Binance:    per-asset 1m futures OHLCV CSVs.
  Source: $BIN_DIR/{ASSET}USDT.csv

By default the loader reads the shipped processed frame data/aligned_15m.parquet,
so these raw sources are optional (see `build_or_load_aligned`).

Output
------
A multivariate 15-minute time series with 5 assets x 4 channels = 20 channels:
    [poly_up, bin_logret, bin_logvol, bin_range] for each of (BTC, ETH, SOL, DOGE, XRP).

Channel index layout (when flat=True): channel j corresponds to
    asset = ASSETS[j // 4]
    feature = FEATURES[j % 4]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ASSETS = ["btc", "eth", "sol", "xrp"]  # DOGE excluded by default → full range from 2026-01-01
ASSETS_WITH_DOGE = ["btc", "eth", "sol", "doge", "xrp"]
FEATURES = ["poly_up", "bin_logret", "bin_logvol", "bin_range"]
NUM_FEATURES_PER_ASSET = len(FEATURES)

# Raw-data directories (only needed to *rebuild* the aligned frame from scratch).
# By default the loader reads the shipped processed cache below, so the raw
# sources are optional; override with the POLY_DIR / BIN_DIR env vars.
POLY_DIR_DEFAULT = os.environ.get("POLY_DIR", "data/polymarket_15m_data")
BIN_DIR_DEFAULT = os.environ.get("BIN_DIR", "data/binance_1m_futures")

# Shipped processed frame: the aligned 16-channel 15-minute series. Loading this
# reproduces every result without needing the raw sources.
ALIGNED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "aligned_15m.parquet")


# ---------------------------------------------------------------------------
# Polymarket loading
# ---------------------------------------------------------------------------

def _load_poly_up_series(asset: str, poly_dir: str) -> pd.Series:
    """Build a 15m series of the UP-token probability for `asset`.

    For each Polymarket market (a single 15m up/down event) we take the LAST
    tick of the UP token before the market's end time, indexed at the market
    end timestamp (which is aligned to 15m boundaries).
    """
    meta = pd.read_parquet(os.path.join(poly_dir, f"{asset}_metadata.parquet"))
    prices = pd.read_parquet(os.path.join(poly_dir, f"{asset}_prices.parquet"))

    meta = meta[meta["closed"]].copy()
    meta["end_dt"] = pd.to_datetime(meta["end_date"], utc=True)
    meta["end_dt"] = meta["end_dt"].dt.floor("15min")

    up = prices[prices["side"] == "up"].copy()
    up["dt"] = pd.to_datetime(up["t"], unit="s", utc=True)
    up = up.sort_values(["market_id", "dt"])
    last_up = up.groupby("market_id", as_index=False).tail(1)[["market_id", "p"]]
    last_up = last_up.rename(columns={"p": "poly_up"})

    merged = meta.merge(last_up, on="market_id", how="inner")
    merged = merged.dropna(subset=["end_dt", "poly_up"])
    merged = merged.drop_duplicates("end_dt", keep="last")
    series = merged.set_index("end_dt")["poly_up"].astype("float32").sort_index()
    series.name = f"{asset}_poly_up"
    return series


# ---------------------------------------------------------------------------
# Binance loading
# ---------------------------------------------------------------------------

def _load_binance_15m(asset: str, bin_dir: str) -> pd.DataFrame:
    """Load Binance 1m OHLCV, resample to 15m, return derived features.

    CRITICAL alignment note (no-leakage):
      We use `closed='right', label='right'` so that the bin labeled `t` contains
      1m bars from `(t-15min, t]` and is OBSERVABLE AT time `t`. This matches
      the Polymarket convention where `poly_up[t]` is the UP-token price for the
      market that ENDS at time `t` (i.e. info from `(t-15min, t]`).

      The previous default (`closed='left', label='left'`) would have made
      `bin_*[t]` cover `[t, t+15min)` — *future* information overlapping the
      next Polymarket market (which resolves at `t+15min`). That produced a
      massive lookahead leak: `corr(bin_logret[t], poly_up[t+1]) ≈ +0.63`.
      With the right-closed convention the same correlation drops to ≈ -0.06
      (just noise).
    """
    path = os.path.join(bin_dir, f"{asset.upper()}USDT.csv")
    df = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()

    agg = df.resample("15min", closed="right", label="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])

    close = agg["Close"].astype("float32")
    out = pd.DataFrame(index=agg.index)
    out[f"{asset}_bin_logret"] = np.log(close / close.shift(1)).astype("float32")
    out[f"{asset}_bin_logvol"] = np.log1p(agg["Volume"].astype("float32"))
    rng = (agg["High"] - agg["Low"]) / agg["Close"].replace(0, np.nan)
    out[f"{asset}_bin_range"] = rng.astype("float32")
    out = out.dropna()
    return out


# ---------------------------------------------------------------------------
# Aligned dataframe
# ---------------------------------------------------------------------------

def build_aligned_dataframe(
    assets: List[str] = ASSETS,
    poly_dir: str = POLY_DIR_DEFAULT,
    bin_dir: str = BIN_DIR_DEFAULT,
) -> pd.DataFrame:
    """Build the wide-format aligned 15m DataFrame for all assets.

    Columns are ordered so that channels for the same asset are contiguous:
        [<asset0>_poly_up, <asset0>_bin_logret, <asset0>_bin_logvol, <asset0>_bin_range,
         <asset1>_..., ...]
    """
    pieces: List[pd.DataFrame] = []
    for a in assets:
        poly = _load_poly_up_series(a, poly_dir).to_frame()
        binf = _load_binance_15m(a, bin_dir)
        merged = poly.join(binf, how="outer")
        # enforce column ordering for this asset
        ordered = [f"{a}_{feat}" for feat in FEATURES]
        merged = merged[ordered]
        pieces.append(merged)

    aligned = pieces[0]
    for p in pieces[1:]:
        aligned = aligned.join(p, how="outer")

    # restrict to intersection time range where all polymarket UP series have data
    poly_cols = [f"{a}_poly_up" for a in assets]
    bin_cols = [c for c in aligned.columns if c not in poly_cols]

    poly_mask = aligned[poly_cols].notna().all(axis=1)
    if poly_mask.any():
        first_valid = aligned.index[poly_mask].min()
        last_valid = aligned.index[poly_mask].max()
        aligned = aligned.loc[first_valid:last_valid]

    # Forward-fill short polymarket gaps (some buckets may miss a tick)
    aligned[poly_cols] = aligned[poly_cols].ffill(limit=4)
    aligned[bin_cols] = aligned[bin_cols].ffill(limit=4)
    aligned = aligned.dropna()
    return aligned


def build_or_load_aligned(
    assets: List[str] = ASSETS,
    poly_dir: str = POLY_DIR_DEFAULT,
    bin_dir: str = BIN_DIR_DEFAULT,
    cache: str = ALIGNED_CACHE,
) -> pd.DataFrame:
    """Return the aligned frame, preferring the shipped processed cache.

    If `cache` exists it is loaded directly (no raw data needed); otherwise the
    frame is rebuilt from the raw Polymarket/Binance sources and cached.
    """
    cols = [f"{a}_{f}" for a in assets for f in FEATURES]
    if cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"cached frame {cache} is missing columns {missing[:4]}")
        return df[cols]
    df = build_aligned_dataframe(assets=assets, poly_dir=poly_dir, bin_dir=bin_dir)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache)
    return df


# ---------------------------------------------------------------------------
# Standardization / splits
# ---------------------------------------------------------------------------

def _time_splits(df: pd.DataFrame, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    n = len(df)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = df.iloc[:n_train]
    valid = df.iloc[n_train : n_train + n_valid]
    test = df.iloc[n_train + n_valid :]
    return train, valid, test


@dataclass
class ScalerStats:
    mean: np.ndarray
    std: np.ndarray
    columns: List[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype("float32")


def fit_standardizer(train_df: pd.DataFrame) -> ScalerStats:
    mu = train_df.mean(axis=0).values.astype("float32")
    sd = train_df.std(axis=0).replace(0, 1.0).values.astype("float32")
    return ScalerStats(mean=mu, std=sd, columns=list(train_df.columns))


# ---------------------------------------------------------------------------
# Torch dataset / loader
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def create_polymarket_loaders(
    seq_len: int = 96,
    pred_len: int = 8,
    batch_size: int = 32,
    assets: List[str] = ASSETS,
    poly_dir: str = POLY_DIR_DEFAULT,
    bin_dir: str = BIN_DIR_DEFAULT,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 0,
) -> Dict[str, object]:
    """Build train/valid/test loaders + scaler + metadata.

    Returns dict with:
      train_loader, valid_loader, test_loader, scaler, columns,
      poly_indices (list of channel indices that are polymarket UP targets),
      df (the aligned dataframe before standardization).
    """
    df = build_or_load_aligned(assets=assets, poly_dir=poly_dir, bin_dir=bin_dir)
    train_df, valid_df, test_df = _time_splits(df, ratios)
    scaler = fit_standardizer(train_df)

    train_np = scaler.transform(train_df.values)
    valid_np = scaler.transform(valid_df.values)
    test_np = scaler.transform(test_df.values)

    train_ds = WindowDataset(train_np, seq_len, pred_len)
    valid_ds = WindowDataset(valid_np, seq_len, pred_len)
    test_ds = WindowDataset(test_np, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    poly_indices = [i for i, c in enumerate(scaler.columns) if c.endswith("_poly_up")]

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "columns": scaler.columns,
        "poly_indices": poly_indices,
        "n_channels": len(scaler.columns),
        "df": df,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--poly-dir", default=POLY_DIR_DEFAULT)
    parser.add_argument("--bin-dir", default=BIN_DIR_DEFAULT)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=8)
    args = parser.parse_args()

    out = create_polymarket_loaders(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=4,
        poly_dir=args.poly_dir,
        bin_dir=args.bin_dir,
    )
    df = out["df"]
    print(f"Aligned dataframe: {df.shape}, range {df.index.min()} -> {df.index.max()}")
    print(f"Columns ({len(out['columns'])}):")
    for c in out["columns"]:
        print(f"  {c}")
    print(f"Polymarket target indices: {out['poly_indices']}")
    print(f"#batches  train={len(out['train_loader'])}  "
          f"valid={len(out['valid_loader'])}  test={len(out['test_loader'])}")

    x, y = next(iter(out["train_loader"]))
    print(f"Sample batch: x={tuple(x.shape)}  y={tuple(y.shape)}  dtype={x.dtype}")
