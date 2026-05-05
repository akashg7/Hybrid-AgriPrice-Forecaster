# 🌾 AgriSense Intelligence Hub
### *The Future of Hybrid Agricultural Forecasting*

[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)](https://github.com/akashg7/Hybrid-AgriPrice-Forecaster)
[![Accuracy](https://img.shields.io/badge/TFT--Accuracy-96.15%25-blue)](https://github.com/akashg7/Hybrid-AgriPrice-Forecaster)
[![Framework](https://img.shields.io/badge/Framework-PyTorch--Lightning-orange)](https://github.com/akashg7/Hybrid-AgriPrice-Forecaster)

---

## 🚀 Overview
**AgriSense AI** is a state-of-the-art hybrid intelligence platform designed to stabilize the agricultural supply chain in India. By combining **Deep Learning (TFT)**, **Gradient Boosting (LGBM)**, and **Computer Vision (CNN)**, we provide farmers and stakeholders with a three-layered advisory system:
1.  **Price Intelligence**: 14-day multi-horizon price forecasting with risk intervals.
2.  **Ecological Intelligence**: Soil-climate synergistic crop recommendations.
3.  **Biological Intelligence**: Interpretability-first plant disease detection.

---

## 🏗️ Architecture
The system follows a **Decoupled Engine Pattern**, ensuring that high-compute forecasting models (TFT) can operate alongside low-latency classifiers (LGBM/CNN).

### Key Components:
- **`Engines/`**: Canonical interface for model inference.
- **`Modules/`**: Specialized technical tracks (Price_TFT, Price_LGBM, Crop_Rec, Disease_Det).
- **`frontend/`**: Next.js dashboard with a premium dark-mode aesthetic.
- **`AgriSense_Hub_Backend.py`**: The central intelligence hub serving real-time APIs.

---

## 📊 Performance Benchmarks
| Module | Model | Primary Metric | Result |
| :--- | :--- | :--- | :--- |
| **Price (Primary)** | **TFT (Epoch 15)** | **Accuracy** | **96.15%** |
| **Price (Secondary)** | **LightGBM** | **SMAPE** | **8.80%** |
| **Crop Recommendation** | **Leaf-wise GBDT** | **F1-Score** | **0.99** |
| **Disease Detection** | **EfficientNet-B0** | **Accuracy** | **98.42%** |

---

## 🛠️ Quick Start

### 1. Backend Hub
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Intelligence Hub
python AgriSense_Hub_Backend.py
```

### 2. Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

---

## 📖 Key Documentation
- **[Executive Summary](./AgriSense_Executive_Summary.md)**: High-level vision and achievements.
- **[Unified Architecture Hub](./AgriSense_Unified_Architecture.html)**: Interactive visual breakdown of all modules.
- **[IEEE Technical Report](./AgriSense_Final_Comprehensive_Report.tex)**: Academic-grade architectural deep-dive.

---
*Developed by the AgriSense AI Team — Bridging Technology and Agriculture.*