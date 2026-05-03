# Module Technical Deep-Dive: Price_TFT
## 🧠 Neural Architecture: Temporal Fusion Transformer

### 1. The Inference Engine
The TFT engine (located in `Engines/TFT_Engine.py`) utilizes a high-capacity Transformer architecture trained on 2+ years of Mandi data.

### 2. Feature Vector Definition (52 Dimensions)
The model consumes a multi-modal vector comprising:
- **8 Known Reals**: `time_idx`, `day_of_year`, `day_of_week`, `month`, and **Fourier Encodings** (`sin1`, `cos1`, `sin2`, `cos2`) to capture cyclical seasonality.
- **44 Unknown Reals**: 
  - **Market Lags**: `modal_lag_1` to `modal_lag_60` (Price trends).
  - **Arrival Dynamics**: `Arrivals`, `arrivals_lag_1/7/30/60` (Supply shocks).
  - **Climate Features**: `temp_avg`, `rainfall`, `humidity`, `solar_radiation` (Crop health proxies).
  - **Engineered Interactions**: `price_x_arrivals`, `temp_anomaly`, `arrival_shock_flag`.

### 3. Mathematical Reasoning
- **Variable Selection Network (VSN)**: This component of the TFT identifies which features are relevant for each time step. For example, during the harvest season, `Arrivals` might be prioritized over `Weather`.
- **Gated Residual Connection (GRN)**: Used throughout the model to suppress noise from non-predictive inputs.
- **Quantile Loss Function**: Optimized to predict the 10th, 50th, and 90th percentiles, providing a probabilistic forecast rather than a simple mean.

### 4. Technical Specs
- **Input Encoder**: 60 Days
- **Forecast Horizon**: 14 Days
- **Scaling**: `GroupNormalizer` with Softplus transformation for non-negative price stability.
- **Weights**: `best-advanced-tft-epoch=07-val_loss=64.02.ckpt`
