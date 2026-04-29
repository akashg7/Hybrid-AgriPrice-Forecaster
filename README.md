# Agricultural Mandi Price Forecasting & Analytics

This project implements a comprehensive Machine Learning pipeline to forecast agricultural commodity prices in Indian Mandis (markets). It utilizes both traditional Gradient Boosting (LightGBM) and state-of-the-art Time Series Foundation Models (Salesforce Moirai) to provide accurate price predictions and market analytics.

## 📊 Project Overview

One Pager - https://www.canva.com/design/DAG02e3n3fY/fw5LU0Ds5T92ALBCZ8ybiQ/edit?utm_content=DAG02e3n3fY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

The goal is to help farmers and traders make informed decisions by predicting **Modal Prices** of various crops. The system offers:
- **Price Forecasting**: 14-day forecast horizon for commodities.
- **Market Recommendations**: identifying the best markets for a specific crop.
- **Analytics**: Historical trends, volatility analysis, and price spike detection.
- **Comparison**: Benchmarking a robust LightGBM regressor against the Zero-Shot Moirai Foundation Model.

## 🧠 Models & Performance

### 1. LightGBM (Gradient Boosting)
This is the **primary recommended model** for this dataset. It simulates a "full-feature" approach using engineered features like price lags, rolling statistics, weather correlations, and calendar effects.

- **Status**: Trained & Tuned suitable for production.
- **Features**: Lagged prices (1, 3, 7, 14, 30 days), Rolling Means/Std, Weather (Temp, Rainfall), Time embeddings.
- **Accuracy (Test Set, 56,826 rows)**:
  - **Accuracy**: **~91.20%** (100 - SMAPE)
  - **MAE**: 160.41 ₹/quintal
  - **SMAPE**: 8.80%
  - *Comparison*: Drastically dominates the global naive baseline (Acc ~21.3%).

### 2. Salesforce Moirai (Foundation Model)
An experimental "Zero-Shot" approach using a pre-trained Transformer model (`moirai-1.1-R-small`) designed for universal time series forecasting.

- **Status**: Experimental / Benchmarking.
- **Approach**: Zero-shot (no fine-tuning on this specific data).
- **Performance**: Currently underperforms the specific LightGBM model (MAE ~543 vs ~310).

### 3. Temporal Fusion Transformer (TFT) — Deep Learning
A multi-horizon forecasting architecture that uses self-attention to capture long-range dependencies and multi-variable interactions (weather + arrivals).

- **Accuracy**: **~86.59%** (14-day horizon, 3,815 validation sequences, epoch 9 checkpoint)
- **MAE**: 319.88 ₹/quintal | **SMAPE**: 13.41%
- **Features**: 30-feature multivariate set (Static, Observed, and Future covariates).
- **Confidence**: Provides probabilistic quantile forecasts for risk management.
- **Note**: Currently underperforms LightGBM on tabular features; further hyperparameter tuning (larger hidden_size, more epochs) would improve results.

### 4. EfficientNet-B0 (Plant Disease Pathology)
A Deep Learning vision track for automated leaf disease diagnosis across 38 classes.

- **Architecture**: EfficientNet-B0 backbone with Two-Phase Transfer Learning.
- **Accuracy**: **98.42%** on test set.
- **Interpretability**: Integrated Grad-CAM to visualize pathological focus areas.

## 📂 Folder Structure

## 🚜 Data Collection & Processing Pipeline

We built a custom pipeline to aggregate data from disparate sources, as no single dataset existed with both price and granular weather/location data.

### 1. Data Scraping (Agmarknet)
We scraped **Agmarknet** (Government of India) to get daily market data.
- **Scope**: Jan 1, 2024 to Dec 1, 2025.
- **Scale**: Daily data for top 10 crops across all Mandis.
- **Challenge**: The data was not available as a ready-made CSV. We had to reverse-engineer the API to fetch data day-by-day.

### 2. Geographic Mapping
Agmarknet provides Mandi names but no coordinates.
- **Solution**: We created a custom mapper to link each District to its Latitude/Longitude.
- **Source**: We utilized the [NASA POWER API](https://power.larc.nasa.gov/) for weather, which requires precise Lat/Lon.
- **Method**: Fuzzy matching district names against a master database of Indian cities.

### 3. Weather Enrichment
Using the coordinate map, we fetched historical weather data from NASA.
- **Features**: Temperature (Max/Min/Avg), Rainfall, Humanity, Solar Radiation, Wind Speed.
- **Logic**: For every single record in our price dataset, we attached the exact weather conditions for that mandi on that specific day.

### 4. Feature Engineering
We expanded the raw dataset from **~20 columns to 70+ columns** to capture deep market signals.

**Key Engineered Features:**
- **Temporal**: `day`, `month`, `year`, `day_of_week`, `is_weekend`, Seasonal Fourier terms (`sin1`, `cos1`).
- **Price Lags**: `modal_lag_1` to `modal_lag_30` (capturing short & long term trends).
- **Rolling Stats**: `rolling_mean` and `rolling_std` windows (volatility indicators).
- **Arrivals**: Lagged arrival volumes and shock features (`arrival_change_7`).
- **Weather Anomalies**: `temp_anomaly`, `rain_anomaly` (deviations from 30-day averages).
- **Spatial**: Latitude/Longitude sine/cosine embeddings.

---

## 🌾 The "Top 10" Crops Strategy

We didn't pick random crops. We selected a curated list of **10 Commodities** that maximize model learning, economic impact, and volatility patterns.

| Crop Category | Crops Selected | Why? |
| :--- | :--- | :--- |
| **High Volatility** | **Onion, Tomato** | Extreme price spikes, politically sensitive, good for anomaly detection. |
| **Staples (MSP)** | **Wheat, Rice** | Massive volume, government regulated, strong seasonal baselines. |
| **Moderate Volatility** | **Potato** | Cold-storage driven, cyclical yearly patterns. |
| **Regional/Spatial** | **Dry Chili, Banana** | Strong regional dependence (e.g., Guntur for Chili), complex supply chains. |
| **Oilseeds** | **Groundnut/Mustard** | Long-term cycles, MSP impact, stable trends. |
| **Seasonal Fruits** | **Watermelon** | Sharp seasonal demand curves, predictable harvest windows. |

This mix ensures our model learns to handle everything from **perishable spikes** (Tomato) to **stable MSP trends** (Wheat).

---

## 📂 Folder Structure

For more details on the datasets and links, see [DATASET_INFO.md](DATASET_INFO.md).

```
AIMLProject/
├── merged_crop_data_with_weather.csv       # Raw source data (Root level)
├── mandi_crop_pairs.csv                    # Metadata of market-commodity pairs
├── run_moirai.py                           # Legacy/Verification script for Moirai
├── data/
│   ├── processed/                          # Engineered Parquet files for ML
│   └── metadata/                           # Series mappings and auxiliary data
├── models/
│   ├── price_lgbm_full_features.pkl        # Best performing LightGBM model
│   ├── moirai_dyn_scaler.pkl               # Scalers for Moirai pipeline
│   └── ...
├── outputs/
│   ├── forecasts/                          # LightGBM forecasts (per mandi/crop)
│   ├── forecasts_moirai/                   # Moirai forecasts
│   └── eval/                               # Detailed evaluation metrics & tables
├── scripts/                                # Logic & Execution Scripts
│   ├── build_training_data.py              # Feature Engineering for ML
│   ├── train_lgbm_full_features.py         # Main training script (LightGBM)
│   ├── prepare_moirai_data.py              # Data formatting for Moirai
│   ├── forecast_all.py                     # Batch forecasting (LightGBM)
│   ├── forecast_all_moirai.py              # Batch forecasting (Moirai)
│   ├── evaluate_lightgbm.py                # Evaluation suite
│   ├── recommend_market.py                 # Analytics: Recommend best market
│   ├── analytics_nearby_markets.py         # Analytics: Find nearby alternatives
│   └── ...                                 # Helpers (list_pairs, etc.)
├── uni2ts/                                 # Clone of Uni2TS library for Moirai
├── DataScraping/                           # Original scraping scripts (legacy)
├── DataPreProcessing/                      # Initial cleaning scripts (legacy)
└── FeatureEngineering /                    # Prototype notebooks/scripts (legacy)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Virtual Environment recommended

### Installation

1.  **Clone/Setup Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt  # (Ensure uni2ts dependencies are installed)
    ```
    *Note: `uni2ts` requires specific torch versions. Refer to `uni2ts` docs if specific install issues arise.*

### How to Run

#### 1. Data Preparation
Converts raw CSV to efficient Parquet format with engineered features.
```bash
# For LightGBM
python scripts/build_training_data.py

# For Moirai (Scaling & formatting)
python scripts/prepare_moirai_data.py
```

#### 2. Train Models
Train the high-accuracy LightGBM model.
```bash
python scripts/train_lgbm_full_features.py
```
*Outputs model to `models/price_lgbm_full_features.pkl` and accuracy to `outputs/eval/`.*

#### 3. Forecasting
Generate 14-day forecasts for all supported Mandi-Commodity pairs.

**Using LightGBM (Recommended):**
```bash
python scripts/forecast_all.py
```

**Using Moirai (Experimental):**
```bash
python scripts/forecast_all_moirai.py
```

#### 4. Analytics & Insights
Use these scripts to query predictions and get reliable insights.
```bash
# Recommend top markets for a specific commodity
python scripts/recommend_market.py --commodity "Tomato" --state "Maharashtra"

# Find nearby markets and compare prices
python scripts/analytics_nearby_markets.py --mandi "Pune" --commodity "Onion" --ref_year 2023

# See best crops for a region
python scripts/best_crop_by_region.py --mode state --name "Gujarat"
```