# 🌾 AgriSense Intelligence Hub: Complete Project Details

This document contains a comprehensive and exhaustive breakdown of the **Hybrid-AgriPrice-Forecaster (AgriSense AI)** project. It details the overarching vision, complete architectural flow, model specifications, features, datasets, and performance evaluations.

---

## 1. Project Vision & Overview
**AgriSense AI** is a state-of-the-art hybrid intelligence platform engineered to stabilize the agricultural supply chain in India. It tackles the severe volatility of agricultural markets by bridging the gap between **Price Prediction**, **Ecological Suitability**, and **Biological Risk**.

The system utilizes a **Decoupled Engine Pattern**, meaning computationally heavy deep learning models (like TFT) operate modularly alongside high-speed classifiers (LGBM) and computer vision modules (CNN). 

The platform offers a three-layered advisory system:
1. **Price Intelligence:** 14-day multi-horizon price forecasting with probabilistic risk intervals.
2. **Ecological Intelligence:** Synergistic soil and climate-based crop recommendations optimized for maximum profitability.
3. **Biological Intelligence:** Interpretability-first plant leaf disease detection for early-warning risk signals.

---

## 2. Technology Stack
* **Frontend:** Next.js (React) with a premium dark-mode aesthetic for the unified user dashboard.
* **Backend Hub:** FastAPI (Python) serving modular AI engines (`AgriSense_Hub_Backend.py`).
* **Deep Learning/Time-Series:** PyTorch, PyTorch Lightning, PyTorch Forecasting.
* **Machine Learning (Tabular):** LightGBM, scikit-learn.
* **Computer Vision:** TensorFlow, Keras (EfficientNet backbone).
* **Data Pipelines & Manipulation:** Pandas, NumPy.
* **External Integrations:** Agmarknet API (reverse-engineered custom scraper), NASA POWER API, IndianCities dataset.

---

## 3. Core Architectural Modules & Models

### A. The "Data-Rich" Forecasting Track (Prices)
This track operates on daily mandi prices and weather data, using a dual-model strategy to balance speed and uncertainty quantification.

#### 1. Primary Engine: Temporal Fusion Transformer (TFT)
* **Status:** The production standard for risk analysis.
* **Purpose:** Multi-horizon (14-day) forecasting providing **Quantile Outputs** (10th, 50th, 90th percentiles). Instead of a single average guess, it gives farmers "worst-case" and "best-case" scenarios.
* **Architectural Details:**
  * **Variable Selection Networks (VSN):** Automatically prunes redundant features.
  * **Gated Residual Networks (GRN):** Allows the network to skip complex pathways if relationships are simple, preventing over-parameterization.
  * **Interpretable Multi-Head Attention:** Focuses on specific long-range temporal dependencies.
* **Hyperparameters (Epoch 15 Heavy-TFT):** `hidden_size=128`, Context Window = 60 days, Dropout = 0.20.
* **Performance (Epoch 15):** 
  * Accuracy: **96.15%** (SMAPE: 3.85%)
  * RMSE: 155.96
  * MAE: 78.89
* **Features Used (70+ Markers):** Fourier Seasonality (sin/cos embeddings), localized volatility (7-day, 30-day lags), price structures, supply shocks (arrivals), and spatial embeddings (lat/lon).

#### 2. Secondary Engine: LightGBM (GOSS-Optimized)
* **Status:** The high-speed benchmark.
* **Purpose:** Deterministic, single-step point prediction. Exceptionally efficient with high-cardinality tabular features (Mandi/District mappings).
* **Performance:**
  * Accuracy: **91.20%**
  * SMAPE: 8.80%
  * MAE: 160.41 INR/qtl
  * Inference Speed: < 100ms per mandi-commodity pair.

#### 3. Experimental Engine: Salesforce Moirai
* **Purpose:** Explored as a zero-shot foundation model using the `uni2ts` library. 
* **Outcome:** Failed to beat domain-specific fine-tuned models (MAE ~543), highlighting the necessity of localized, covariate-heavy training.

### B. The "Ecological" Intelligence Track
Bridging agronomic viability with economic yield.

#### 1. Crop Recommendation Engine
* **Purpose:** Determines the best crop to grow given localized soil/weather, and crosses it with 14-day price forecasts to recommend the most *profitable* option.
* **Algorithm:** Leaf-wise Gradient Boosting Decision Tree (LightGBM / GBDT).
* **Dataset:** 45 composite soil-climate markers (N, P, K, pH, Temp, Rainfall) sourced from Kaggle.
* **Performance:** F1-Score: **0.99**

### C. The "Biological" Intelligence Track
Acting as a biological risk radar via visual diagnostics.

#### 1. Plant Disease Detection Engine
* **Purpose:** Automated plant disease detection from leaf images.
* **Architecture:** **EfficientNet-B0** (Transfer Learning) with a custom classification head.
* **Why EfficientNet:** Uses Compound Scaling (Width, Depth, Resolution) and MBConv (Mobile Inverted Bottleneck) to be incredibly parameter-efficient (~5.3MB) while maintaining state-of-the-art accuracy. Suitable for edge deployment.
* **Training Method:** Two-phase transfer learning (frozen feature extraction -> unfrozen top-layer fine-tuning).
* **Interpretability:** Integrates **Grad-CAM** heatmaps to visualize the exact diseased lesions the model focuses on, offering complete transparency.
* **Dataset:** 87,000+ images across 38 classes (Merged from two Kaggle datasets with perceptual hashing for deduplication).
* **Performance:**
  * Accuracy: **98.42%**
  * Precision/Recall/F1: ~98.45%
  * Healthy vs Diseased Recall: ~99.8%.

---

## 4. Data Processing Pipelines & Engineering

### 1. Ingestion / Data Collection
* **Market Prices:** Scraped daily from **Agmarknet** (Jan 2024 - Dec 2025) via a custom Python scraper (`DataScraping/CropData/fast_scrape.py`). Target covers top 10 volatile commodities (Onion, Tomato, Wheat, Rice, Mustard, etc.).
* **Geospatial Coordinates:** `generate_all_coordinates.py` fuzzy-matches Mandi districts to IndianCities GitHub database to assign accurate Lat/Longs.
* **Weather Enrichment:** `weather_collector.py` queries **NASA POWER API** using the coordinates to fetch deep meteorological data (T2M, T2M_MAX, PRECTOTCORR, RH2M, Solar Radiation, Wind Speed).

### 2. Feature Engineering
Raw data (~20 columns) is massively expanded to 70+ variables to capture market psychology:
* **Temporal:** `day`, `month`, `year`, `is_weekend`, Fourier cyclical terms (`sin1`, `cos1`).
* **Price Dynamics:** 1, 3, 7, 14, 30-day lags. Rolling means/stds. 7/30-day Volatility (`zscore_7`, `momentum_7`).
* **Supply:** Arrival lags, average arrivals, arrival shocks.
* **Environmental Anomaly:** Temperature and rain deviations from the 30-day local mean.
* **Spatial:** Geographic proximity embeddings (`lat_sin`, `lon_cos`).

---

## 5. Directory & File Structure Details

The project is highly modular. Here is a map of the repository:

```
Hybrid-AgriPrice-Forecaster/
├── Engines/                     # Canonical abstraction interface for model inference.
│   ├── TFT_Engine.py
│   ├── LGBM_Engine.py
│   ├── Crop_Engine.py
│   └── Disease_Engine.py
├── Modules/                     # The encapsulated weights and dedicated data for each AI track
│   ├── Price_TFT/               # Holds checkpoints for the Heavy TFT Model
│   ├── Price_LGBM/              # Holds the LightGBM .pkl models
│   ├── Crop_Rec/                # Holds the Kaggle trained GBDT crop models
│   └── Disease_Det/             # Contains EfficientNet-B0 weights
├── MandiPricePredictionSystem/  # Core R&D for Price Tracking
│   ├── DataScraping/            # Scraping logic for Agmarknet and NASA Power
│   ├── FeatureEngineering/      # Dimensionality expansion scripts
│   ├── ARCHITECTURE.md          # Architectural flow of the data
│   ├── DATASET_INFO.md          # Exhaustive data sourcing notes
│   ├── TFT_RESULTS.md           # Breakdown of epoch 9 vs epoch 15 evaluation
│   └── scripts/                 # Training scripts (tft_multivariate_training.py, evaluate_lightgbm.py)
├── CropRecommendationSystem/    # R&D for Ecological crop choices
│   ├── profitable_crop_recommender.py
│   └── train_crop_model.py
├── plantDiseasesDetection/      # R&D for Computer Vision disease tracking
│   ├── notebooks/               # Main Colab pipeline (crop_disease_detection.ipynb)
│   ├── src/                     # Data loading, augmentation, and Grad-CAM implementation
│   └── README.md                # Extensive details on the 38-class dataset
├── report/                      # Academic deliverables
│   ├── TFT_and_PlantDisease_StudyGuide.md # Viva-defense guide, literature references
│   └── main.tex                 # Academic latex report
├── frontend/                    # Next.js User Interface
├── AgriSense_Hub_Backend.py     # FastAPI server orchestrating the `Engines/` directory
├── AgriSense_Executive_Summary.md # High-level business overview
├── Master_Project_Report.md     # Production technical summary
└── README.md                    # Setup and entrypoint instructions
```

---

## 6. End-to-End System API Flow (`AgriSense_Hub_Backend.py`)
1. FastAPI launches and dynamically loads weights into the `Engines/` objects.
2. Evaluates the `hierarchy_map.csv` to ensure only valid Mandi-Commodity pairs (minimum 74 rows of historical data) are exposed to the UI.
3. Exposes unified endpoints:
   - `POST /api/tft/predict`
   - `POST /api/lgbm/predict`
   - `POST /api/crop/recommend`
   - `POST /api/disease/detect`
4. The Next.js frontend queries these endpoints seamlessly, abstracting the ML complexity from the user.

---

## 7. Future Strategic Roadmap
- **Hyper-Local Expansion:** Integration of direct village-level weather station IoT data.
- **Continuous Learning:** Streaming LightGBM updates without triggering full-dataset retraining loops.
- **Edge Deployment:** Implementing model quantization on EfficientNet-B0 to allow for offline mobile inference in remote farming regions.
