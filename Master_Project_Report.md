# Technocratic Master Report: AgriSense Intelligence Hub
**Developed by: Karthik Reddy (230035) & Akash G (230098)**
## 🚀 Engineering Roadmap: Verified Project Evolution

### 1. The Multi-Track Forecasting Strategy
The project implements a dual-track forecasting architecture to balance speed and temporal depth:

- **Track A: LightGBM (Full Features)** — **THE BENCHMARK**
  - **Accuracy (SMAPE)**: **91.20%**
  - **MAE**: **160.41 INR/qtl**
  - **Reasoning**: Evaluated on 56,826 test rows, LightGBM remains the strongest point-forecaster due to its superior handling of high-cardinality tabular features (Mandi, Commodity) and efficient leaf-wise growth.
  
- **Track B: Temporal Fusion Transformer (TFT)** — **THE SEQUENCE MODEL**
  - **Accuracy (SMAPE)**: **86.59%**
  - **Horizon**: **14-Day Multi-Step**
  - **Reasoning**: While point-accuracy trails LGBM, TFT provides **Probabilistic Forecasting** (Quantiles P10, P50, P90) and uses **Self-Attention** to identify long-range dependencies in market shocks.

### 2. Feature Engineering Pipeline (The Truth)
Our models are powered by a rigorous preprocessing pipeline verified from `build_dl_features_fast.py`:

- **Static Covariates**: `Mandi`, `Commodity` embeddings.
- **Known Reals**: `time_idx`, `day_of_year`, `sin1`, `cos1` (Fourier encoding for seasonality).
- **Unknown Observed**: `target_price`, `temp_avg`, `humidity`, `rainfall`, `rolling_mean_7`, `volatility_7`, `momentum_7`.
- **Reasoning**: We use a 30-day encoder length to capture recent market volatility and a 14-day decoder for short-term planning.

### 3. Plant Disease Diagnosis (Computer Vision)
- **Architecture**: **EfficientNet-B0** (Transfer Learning)
- **Dataset Scale**: **87,000+ images** across **38 classes**.
- **Accuracy**: **98.42%**
- **Reasoning**: EfficientNet-B0 was selected for its **Compound Scaling** (Width, Depth, Resolution), allowing it to achieve state-of-the-art results on the PlantVillage benchmark with only 5.3MB in weights.

---

### 📈 Presentation Content (PPT Structure)

#### Slide 1: The Vision
- AgriSense: A decision-support engine for farmers and market analysts.
- Core Pillars: Probabilistic Price Forecasting, Ecological Matching, Vision Diagnostics.

#### Slide 2: Price Forecasting (LGBM vs. TFT)
- **LightGBM**: 91.2% Accuracy on single-step prediction.
- **TFT**: 14-Day horizon with Quantile confidence bands.
- Show: MAE 160.4 vs 319.9.

#### Slide 3: Disease Radar (EfficientNet-B0)
- 98.4% Precision across 38 pathologies.
- Real-time diagnostics with Grad-CAM interpretability.

#### Slide 4: Data Engineering (Preprocessing)
- 250k+ Rows of Mandi Data.
- 30-Day sequence lookback.
- Integration of IMD Weather data and Agmarknet Price logs.

---

### 🎓 Evaluator Deep-Dive (Verified Q&A)

**Q: Why does LightGBM outperform your Deep Learning model (TFT)?**
*A: LightGBM is exceptionally good at tabular regression. The TFT is tasked with a much harder problem: a 14-day multi-step horizon with sequence modeling. While TFT's point accuracy is lower, it provides uncertainty quantification (Quantiles) which is more valuable for risk-based decision making.*

**Q: Why EfficientNet-B0?**
*A: It uses Compound Scaling to optimize resolution and depth together. This allows us to detect tiny lesion patterns (rust, blight) with 98.4% accuracy while keeping the model small enough for mobile deployment.*

**Q: How do you handle seasonality?**
*A: We use Fourier Transforms (Sin/Cos) of the calendar date. This creates a circular coordinate system so the model understands December and January are chronologically adjacent.*
