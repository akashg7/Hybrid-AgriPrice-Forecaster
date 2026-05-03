# Module 2: Price Forecasting (LGBM)
## 🌳 Intelligence Profile: Gradient Boosted Trees

### 1. Overview
The LightGBM (Light Gradient Boosting Machine) module provides high-speed, high-accuracy tabular regression. It serves as our efficient baseline, capable of processing thousands of market points in milliseconds using leaf-wise tree growth.

### 2. Technical Specifications
- **Model**: LightGBM (Gradient Boosting Decision Tree)
- **Feature Set**: 16 Core Tabular Features
- **Learning Rate**: 0.05
- **Boosting Type**: GBDT
- **Accuracy**: 91.2%

### 3. Study Guide & Core Concepts
- **Gradient Boosting**: An ensemble technique where each new tree corrects the errors (residuals) of the previous trees.
- **Leaf-wise Growth**: LGBM grows trees vertically (leaf-wise) rather than horizontally (level-wise), which results in lower loss and higher accuracy on large agricultural datasets.
- **Feature Importance**: We use 'Gain' and 'Weight' metrics to rank features like 'Lagged Modal Price' and 'Arrival Volume'.

### 4. Expected Viva Questions
- **Q: Why use LGBM if you already have TFT?**
  - *A: LGBM is computationally cheaper and faster for real-time edge deployment. It also serves as a critical baseline to prove that our complex TFT model is actually adding value over traditional methods.*
- **Q: What are 'Lagged Features'?**
  - *A: These are past values of the target variable (e.g., Price 1 day ago, 7 days ago). They are the most influential predictors in tree-based market models.*
- **Q: How does LGBM handle outliers in Mandi arrivals?**
  - *A: It uses histogram-based binning which reduces the impact of extreme outliers compared to standard XGBoost.*

---
## 📑 Evaluation Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **RMSE** | 160.4 | Superior on short-term 1-day horizons |
| **MAPE** | 8.8% | High precision for stable commodities |
| **Training Time** | < 2s | Extremely efficient for retraining |
