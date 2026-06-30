# AgriSense Intelligence Hub: Production Technical Report
**Developed by: Karthik Reddy (230035) & Akash G (230098)**

## 🚀 System Overview: Integrated Agricultural Intelligence
AgriSense AI is a unified decision-support platform designed to stabilize agricultural incomes by bridging the gap between market volatility and ecological suitability. The system utilizes a **Decoupled Engine Pattern**, allowing for modular inference across price forecasting, crop recommendation, and disease diagnostics.

### 1. The Production Forecasting Strategy
The platform implements a dual-track forecasting architecture to provide both point-precision and risk quantification:

- **Track A: Temporal Fusion Transformer (TFT)** — **THE PRODUCTION STANDARD**
  - **Accuracy (SMAPE)**: **96.15% (3.85%)**
  - **RMSE**: **155.96**
  - **MAE**: **78.89**
  - **Momentum Signal**: **+1.0** (High Confidence)
  - **Horizon**: **14-Day Multi-Step**
  - **Architectural Scaling**: 
    - **Hidden Size**: Increased from 16 to **128** (8x capacity).
    - **Context Window**: Expanded from 14 to **60 days** (Deep lookback).
    - **Dropout**: Optimized to **0.20** for generalization.
    - **Training**: Finalized at **Epoch 15**.
  - **Reasoning**: By utilizing Self-Attention and Gated Residual Networks, the TFT identifies long-range dependencies and supply shocks, providing probabilistic quantile corridors for risk analysis.
  
- **Track B: LightGBM (GOSS-Optimized)** — **THE BENCHMARK**
  - **Accuracy (SMAPE)**: **91.20%**
  - **Inference Speed**: < 100ms per mandi-commodity pair.
  - **Reasoning**: Handles high-cardinality tabular features (Mandi/District mapping) with extreme efficiency, serving as the high-speed point-forecaster for real-time dashboard interactions.

### 2. High-Dimensional Ecological Intelligence
- **Module**: Crop Recommendation Engine
- **Accuracy (F1-Score)**: **0.99**
- **Feature Engineering**: 45 composite soil-climate markers, including nutrient ratios ($N:P:K$), heat indices, and aridity proxies.
- **Unified Logic**: Biologically viable crops are filtered through the 14-day price forecasting track to recommend only the most profitable cultivation paths.

### 3. Biological Risk Radar (Computer Vision)
- **Architecture**: **EfficientNet-B0** (Transfer Learning)
- **Dataset Scale**: **87,000+ images** across **38 classes**.
- **Accuracy**: **98.42%**
- **Interpretability**: Integrated **Grad-CAM** heatmaps to visualize biological triggers, providing transparency for agronomist auditing.

---

### 🛠 Architecture: The Decoupled Hub
The system is deployed via a **Next.js Production Dashboard** communicating with a **FastAPI Modular Hub**.

- **Modular Engines**: All models are encapsulated in a canonical `Engines/` interface, allowing for model weights to be updated (e.g., from Epoch 9 to Epoch 15) without changing the API logic.
- **Data Fusion**: Automated Agmarknet scraping and NASA POWER API integration ensure the models are always served with the latest meteorological and market data.

### 📊 Performance Summary
| Module | Model | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Price Forecasting** | TFT (Epoch 15) | Accuracy | **96.15%** |
| **Price Forecasting** | LightGBM | SMAPE | **8.80%** |
| **Crop Recommendation** | GBDT | F1-Score | **0.99** |
| **Plant Disease** | EfficientNet-B0 | Accuracy | **98.42%** |

---

### 🎓 Technocratic Assessment
**Q: Why does the system use a hybrid of TFT and LightGBM?**
*A: While LightGBM is faster for deterministic point-prediction, the TFT provides essential uncertainty quantification (Quantiles). In agriculture, knowing the 'worst-case' price (P10) is often more valuable than a single average prediction.*

**Q: How is data noise handled?**
*A: We use a 30-day sequence lookback and rolling volatility markers. This dampens short-term sensor or reporting noise while allowing the model to stay sensitive to genuine market momentum.*

**Q: What is the benefit of the Decoupled Architecture?**
*A: It allows the system to scale. New models (e.g., foundation models like Moirai) can be added as new Engines in the Hub without requiring a frontend or API rewrite.*
