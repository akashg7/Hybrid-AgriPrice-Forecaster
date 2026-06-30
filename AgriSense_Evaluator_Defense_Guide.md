# AgriSense AI: Strategic Defense & Evaluator Q&A Guide
**Project Defense: B.Tech Computer Science & AI (2024-2026)**

This guide provides technocratic justifications and literature-backed responses for common evaluator questions regarding the AgriSense AI architecture, methodology, and design choices.

---

## 1. Literature Review & Model Selection

**Q: Why use a Temporal Fusion Transformer (TFT) instead of standard LSTMs?**
*   **Technocratic Answer:** LSTMs struggle with long-range dependencies and cannot natively handle heterogeneous data types (static vs. dynamic). The TFT utilizes **Gated Residual Networks (GRN)** to skip irrelevant inputs and a **Variable Selection Network (VSN)** to automatically rank feature importance at each time step.
*   **Literature Link:** *Lim et al. (2021)* demonstrated that TFTs outperform LSTMs in multi-horizon forecasting by explicitly modeling temporal patterns with self-attention while maintaining interpretability.

**Q: Why do you still keep LightGBM if you have a Deep Learning model (TFT)?**
*   **Technocratic Answer:** This is a **Hybrid Ensemble strategy**. While TFT provides superior **probabilistic quantile forecasts** (risk analysis), LightGBM remains the state-of-the-art for **point-precision on tabular data**. Tree-based models handle high-cardinality categorical variables (like specific Mandi IDs) more efficiently than neural networks in low-latency scenarios.
*   **Literature Link:** *Grinsztajn et al. (2022)* "Why do tree-based models still outperform deep learning on typical tabular data?" justifies using GBDTs as a primary benchmark.

**Q: What is the significance of the Moirai (Zero-Shot) comparison?**
*   **Technocratic Answer:** It highlights the **"Localization Gap"**. Foundation models like Moirai are pre-trained on global trends but fail to capture the hyper-local "Black Swan" events of Indian agricultural markets (e.g., sudden regional hailstorms or unseasonal rainfall). Our 96.15% accuracy on TFT vs. Moirai's poor performance proves that **localized domain-specific fine-tuning** is non-negotiable for agricultural intelligence.

---

## 2. System Architecture & Design

**Q: Explain the "Decoupled Engine Pattern" you implemented.**
*   **Technocratic Answer:** We decoupled the **Inference Logic** from the **Model Weights**. All models (TFT, LGBM, EfficientNet) are encapsulated in standalone `Engine` classes in a canonical `Engines/` directory. This allows the backend to update weights (e.g., from Epoch 9 to Epoch 15) without changing a single line of API code. It ensures **Production Scalability** and **Hot-swappability**.

**Q: How does the system handle "Black Swan" events not present in historical data?**
*   **Technocratic Answer:** While purely historical models are reactive, we mitigate this through **Exogenous Weather Fusion**. By integrating real-time NASA POWER telemetry (rainfall, temp anomalies), the model "sees" the biological trigger (e.g., a storm) 7-14 days before it manifests as a supply shock/price surge in the Mandi.

**Q: Why use EfficientNet-B0 for disease detection instead of ResNet-50?**
*   **Technocratic Answer:** **Compound Scaling efficiency**. EfficientNet-B0 uses 1/10th the parameters of ResNet-50 while achieving higher accuracy on the PlantVillage dataset (98.42%). This allows for **low-latency edge deployment** and real-time inference on the dashboard without requiring high-VRAM GPUs.

---

## 3. Data Engineering & Feature Logic

**Q: Why compute 70+ features? Isn't that prone to overfitting?**
*   **Technocratic Answer:** We prevent overfitting through **GOSS (Gradient-based One-Side Sampling)** in LightGBM and **Dropout (0.20)** in TFT. The high feature density is necessary to capture **non-linear interactions** between market arrivals and weather (e.g., Rainfall only spikes prices if the crop is in its peak arrival window).

**Q: How did you solve the "Noisy Spatial Label" problem in Agmarknet?**
*   **Technocratic Answer:** We implemented **Normalized Levenshtein Distance** algorithms to map misspelled Agmarknet district names to a master geospatial database. This allowed us to retrieve precise NASA weather coordinates which was the critical missing link in previous research.

---

## 4. Key Performance Benchmarks (The Defense Numbers)

| Metric | Value | Justification |
| :--- | :--- | :--- |
| **TFT Accuracy** | **96.15%** | Achieved through 128-unit Hidden State & 60-day context. |
| **SMAPE** | **3.85%** | Symmetric error handling for volatile commodity prices. |
| **RMSE** | **155.96** | 75% reduction in error variance from prototype to production. |
| **Disease Accuracy**| **98.42%** | Validated via Grad-CAM interpretability heatmaps. |
| **Crop F1-Score** | **0.99** | Derived from 45 high-dimensional soil-climate markers. |

---

## 5. Potential "Trap" Questions & Responses

**Q: 99% accuracy on crop recommendation seems too good to be true. Why?**
*   **Correct Response:** "You are correct to note the high precision. This is a result of the **noiseless nature of the Kaggle benchmark dataset**. In a real-world scenario with sensor noise, we expect a degradation of 5-10%. However, our focus was on demonstrating that **feature interaction engineering** (e.g., Temperature $\times$ Humidity index) significantly outperforms raw input baselines."

**Q: Why 14 days? Why not 30 or 60?**
*   **Correct Response:** "The 14-day horizon was selected to align with the **perishability cycle** of Indian vegetables and the standard planning window for smallholder farmers. Beyond 14 days, the signal-to-noise ratio in decentralized markets decays significantly, making the forecast less actionable for immediate market timing."
