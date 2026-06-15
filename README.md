# PRISM — Reproducibility Code

Code accompanying *"PRISM: A Parameter-Efficient Multi-Modal Benchmark for
Polymarket 15-Minute Up/Down Crypto Prediction Markets"* (Experiment, Analysis &
Benchmark track). This repository is self-contained: it ships the model, all
in-repo baselines, the processed dataset, and the per-seed result files, so every
table and figure in the paper can be regenerated.

PRISM (**P**rediction-market **R**egime-aware **I**ntegrated **S**patiotemporal
**M**odel) is a regime-aware hypernetwork-of-experts: a regime encoder, a dynamic
cross-asset graph, a multi-scale frequency bank, and a low-rank hyper-linear core.

## Repository layout
```
prism/                  PRISM model (regime encoder, dyn. graph, freq bank, hyper-linear)
baselines/              vendored baselines: Linear/DLinear/NLinear/RLinear,
                        PatchTST, iTransformer, Autoformer, FEDformer
groupc/                 vendored domain baselines: StockMixer, SAMBA
data/
  aligned_15m.parquet   processed 16-channel 15-min frame (21,669 rows) — loads by default
runs/                   per-seed result CSV/JSON (regenerate tables without retraining)
model_zoo.py            build_model(name, n_ch, seq_len, pred_len) factory + ALL_MODELS
data_loader.py          aligned loader, temporal split, train-only standardization
training.py             shared train/eval loop (AdamW, cosine, early stopping)
benchmark.py            MAE/RMSE leaderboard + naive floors
outcome_metrics.py      proper scoring rules (Brier / log-loss / ROC-AUC) + calibration
delong_test.py          significance: DeLong, permutation, timestamp-block bootstrap
perasset_psr.py         per-asset Brier / log-loss / ROC-AUC
ablation.py             component ablation
outoftime.py            out-of-time stability over test sub-windows
pnl_backtest.py         economic backtest (PnL net of costs)
classify.py             cross-entropy-head control
extra_baselines.py      input-conditioning controls: MoLE, AdaLinear
forwardfill_audit.py    forward-fill leakage check          (needs raw data, see below)
audit_leakage.py        temporal-alignment leakage audit    (needs raw data, see below)
tslib_models.py         wrappers for Time-Series-Library baselines (external, see below)
```

## Environment
```bash
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```
Python ≥ 3.10, PyTorch ≥ 2.0. Results in the paper were produced with
torch 2.5.1 + CUDA 12.1 on a single NVIDIA RTX 4090. A CUDA GPU is recommended
but not required (training falls back to CPU).

## Data
The processed, temporally-aligned frame `data/aligned_15m.parquet` (4 assets ×
4 features = 16 channels, 21,669 fifteen-minute buckets) is **shipped** and loaded
automatically — no raw data or downloads are needed to reproduce the results.

Two scripts (`forwardfill_audit.py`, `audit_leakage.py`) reconstruct artifacts
from the **raw** sources and therefore need them. Point the loader at the raw
directories with environment variables and the frame will be rebuilt:
```bash
export POLY_DIR=/path/to/polymarket_15m_data     # Polymarket 15-min UP-token parquet
export BIN_DIR=/path/to/binance_1m_futures        # Binance USDT-M 1-min OHLCV CSV
```

## Quickstart (sanity check)
```bash
python -c "from data_loader import create_polymarket_loaders as L; b=L(); \
print('channels', b['n_channels'], 'poly targets', b['poly_indices'])"
```

## Reproducing the paper
All learned models use a temporal 70/15/15 split, train-only standardization,
AdamW + cosine schedule, gradient clipping, 15 epochs with early stopping
(patience 5) on validation MSE, and seed 2026 (or seeds 2024/2025/2026 for
multi-seed runs). PRISM uses lr = 8e-4; other models lr = 1e-3.

| Paper item | Command |
|------------|---------|
| Full MAE/RMSE leaderboard + naive floors | `python benchmark.py --models "$(python -c 'from model_zoo import ALL_MODELS;print(",".join(ALL_MODELS))')" --epochs 15` |
| Three-seed subset | `python benchmark.py --models "PRISM,LSTM,TSMixer,Linear,Transformer,RLinear" --seed 2026` (repeat 2024/2025) |
| Proper scoring rules (Brier/log-loss/AUC) + calibration | `python outcome_metrics.py --seeds 2024,2025,2026 --epochs 15` |
| Significance (DeLong / permutation / block bootstrap) | `python delong_test.py --seeds 2024,2025,2026 --epochs 15` |
| Per-asset proper scoring rules | `python perasset_psr.py --seeds 2024,2025,2026 --epochs 15` |
| Component ablation | `python ablation.py --seeds 2024,2025,2026 --epochs 15` |
| Out-of-time stability | `python outoftime.py --seeds 2024,2025,2026 --epochs 15` |
| Economic backtest (PnL net of costs) | `python pnl_backtest.py --seeds 2024,2025,2026 --epochs 15` |
| Cross-entropy-head control | `python classify.py --seeds 2024,2025,2026 --epochs 15` |
| Forward-fill leakage check *(needs raw data)* | `python forwardfill_audit.py --seeds 2024,2025,2026` |

Each script writes a CSV/JSON under `runs/`. The `runs/` directory already
contains our outputs, so the summary tables can be re-derived without retraining.
Approximate single-run training time on an RTX 4090: PRISM 87 s, LSTM 15 s,
linear family 34–60 s, FEDformer ~344 s.

## Extended baselines (Time-Series-Library)
TimeXer, TimeMixer, TSMixer, TimesNet, and SegRNN are loaded from an external
clone of the Time-Series-Library:
```bash
git clone https://github.com/thuml/Time-Series-Library
export TSLIB_PATH=/path/to/Time-Series-Library
```
The remaining 13 models (PRISM, the linear family, PatchTST, iTransformer,
Autoformer, FEDformer, LSTM, Transformer, StockMixer, SAMBA, MoLE, AdaLinear) are
in-repo and need no external code.

## Notes
- Evaluation is reported on the four `poly_up` (UP-probability) target channels in
  `[0,1]`; the binary outcome label is the sign of the realized log-return.
- Training minimizes MSE over all 16 channels; the primary metrics are proper
  scoring rules on the binary resolution.
