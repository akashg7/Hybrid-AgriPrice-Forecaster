# Temporal Fusion Transformer — Verified Results & Analysis

## 1. Checkpoint Details

| Property | Value |
|---|---|
| **File** | `epoch=9-step=43310.ckpt` (4.08 MB) |
| **Epoch** | 9 (best val_loss at epoch 6) |
| **Steps** | 43,310 |
| **Parameters** | 54,029 (all non-zero) |
| **Architecture** | hidden_size=16, 1 attention head, dropout=0.1 |
| **Loss** | QuantileLoss (7 quantile outputs) |
| **Framework** | PyTorch Forecasting + Lightning |

## 2. Evaluation Results

Evaluated using `scripts/evaluate_tft.py` — inference only, no retraining.

### Regime B: 14-Day Multi-Horizon (TFT evaluation)

| Model | MAE (₹/qtl) | RMSE | MAPE | SMAPE | Accuracy |
|---|---:|---:|---:|---:|---:|
| Naive (per-group last-value) | 344.36 | 799.32 | 15.53% | 13.14% | 86.86% |
| **TFT (epoch 9)** | **319.88** | **647.88** | **15.77%** | **13.41%** | **86.59%** |

### Regime A: Single-Step (LightGBM evaluation)

| Model | MAE (₹/qtl) | RMSE | MAPE | SMAPE | Accuracy |
|---|---:|---:|---:|---:|---:|
| Naive (global constant) | 1,284.38 | 1,757.20 | 55.19% | 78.66% | 21.34% |
| **LightGBM** | **160.41** | **355.80** | **10.77%** | **8.80%** | **91.20%** |

> **Why different naive baselines?** LightGBM's naive uses one global constant for all mandis/crops (weak baseline → 21%). TFT's naive uses the last known price per mandi-commodity pair (strong baseline → 87%). These regimes measure different things and should not be mixed in one table.

### Key Finding

TFT does NOT beat the per-group naive baseline on SMAPE (13.41% vs 13.14%). It improves MAE by 7.1% and RMSE by 18.9%, meaning it handles outliers better but has no advantage on average relative error. The model shows signs of both **underfitting** (can't beat naive) and **overfitting** (epoch 9 slightly worse than epoch 6).

## 3. Why This Happened — Analysis

| Factor | Detail | Literature |
|---|---|---|
| **Tiny architecture** | hidden_size=16, 1 attention head — too small for 4,612 groups | Lim et al. (2021) use hidden_size=160+ |
| **Price autocorrelation** | Mandi prices are sticky; naive carry-forward is inherently strong | Box et al. (2015) — Time Series Analysis |
| **Bias-variance tradeoff** | Model too small for signal, big enough for noise | Geman et al. (1992) — Neural Networks and the Bias/Variance Dilemma |
| **Multi-horizon difficulty** | Predicting 14 days at once is harder than 1-step-ahead | Taieb & Hyndman (2014) — Multi-step forecasting |
| **Tree advantage on tabular** | Gradient boosting often beats DL on feature-engineered data | Grinsztajn et al. (2022, NeurIPS) |

## 4. Literature Review (Gist)

### Core TFT Paper
**Lim et al. (2021)** — *"Temporal Fusion Transformers for interpretable multi-horizon time series forecasting"*, Int. J. Forecasting.
- Introduced Variable Selection Networks, Gated Residual Networks, and interpretable attention for heterogeneous time series.
- Recommended hidden_size=160-240 for production — our 16 is 10x smaller.

### Why Tree Models Win on Tabular Data
**Grinsztajn et al. (2022)** — *"Why do tree-based models still outperform deep learning on typical tabular data?"*, NeurIPS.
- On medium-sized tabular datasets with engineered features, gradient boosting (LightGBM/XGBoost) consistently matches or beats deep learning.
- Our result (LightGBM 91.2% vs TFT 86.6%) is consistent with this finding.

### Zero-Shot Foundation Models
**Woo et al. (2024)** — *"Unified training of universal time series forecasting transformers"* (Moirai), ICML.
- Zero-shot models struggle on localized, covariate-heavy domains.
- Our Moirai result (MAE ~543) confirms this — domain-specific training is essential.

### Naive Baseline Strength
**Makridakis et al. (2018)** — *"Statistical and Machine Learning forecasting methods: Concerns and ways forward"*, PLOS ONE.
- Simple methods (naive, exponential smoothing) often outperform complex ML models on real-world time series.
- Our finding (naive ≈ TFT) aligns with this well-documented phenomenon.

### Agricultural Price Volatility
**Chand (2012)** — *"Development policies and agricultural markets"*, Economic and Political Weekly.
**NITI Aayog (2019)** — *Demand and Supply Projections Towards 2033*.
- Indian agricultural prices exhibit extreme volatility (200-400% swings in perishables).
- Information asymmetry costs ~$8B annually in post-harvest losses.

## 5. What TFT Provides That LightGBM Cannot

Despite lower accuracy, TFT offers unique capabilities:

1. **Probabilistic forecasts** — 7 quantile outputs (confidence bands) vs single point
2. **Multi-horizon coherence** — all 14 days predicted simultaneously
3. **Interpretability** — Variable Selection Networks + temporal attention weights
4. **Sequence awareness** — captures patterns across the 30-day encoder window

## 6. What Would Improve TFT

- `hidden_size=128` or `256` (current: 16)
- `attention_head_size=4` (current: 1)
- Train 30+ epochs on GPU with proper LR scheduling
- Larger encoder window (60-90 days)
- Add more observed covariates (arrivals, all weather vars)
