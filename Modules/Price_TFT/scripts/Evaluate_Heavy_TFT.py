"""
Evaluate_Heavy_TFT.py — High-precision verification of the Heavy TFT model.
Compares 14-day forecasts against actual market prices from the validation set.
"""

import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data/features.csv"
CKPT_PATH = BASE_DIR.parent / "models/tft-heavy-model/best-advanced-tft-epoch=15-val_loss=44.81.ckpt"

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100
    
    # Directional Accuracy
    # Did we correctly predict if price moves UP or DOWN?
    # We compare the change from the last historical point
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "Accuracy": 100 - smape}

def main():
    print("="*60)
    print("  AGRISENSE HEAVY-TFT EVALUATION ENGINE")
    print("="*60)

    # 1. Load Data
    print(f"📂 Loading data: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Mandi", "Commodity", "date"])
    
    # Sync time_idx and missing values
    df["time_idx"] = df.groupby(["Mandi", "Commodity"]).cumcount()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby(["Mandi", "Commodity"])[numeric_cols].ffill().fillna(0)
    df["target_price"] = df["target_price"].clip(lower=1.0)
    
    max_prediction_length = 14
    max_encoder_length = 60
    # Use the same cutoff as training
    training_cutoff = df["time_idx"].max() - 30

    # 2. Reconstruct Dataset
    print("🛠️ Reconstructing Dataset structure...")
    known_reals = ["time_idx", "day_of_year", "day_of_week", "month", "sin1", "cos1", "sin2", "cos2"]
    unknown_reals = [
        "Arrivals", "temp_avg", "temp_max", "temp_min", "rainfall", "humidity", "solar_radiation", "wind_speed",
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14", "modal_lag_30", "modal_lag_60",
        "arrivals_lag_1", "arrivals_lag_7", "arrivals_lag_30", "arrivals_lag_60",
        "rolling_mean_7", "rolling_mean_30", "rolling_mean_60",
        "rolling_std_7", "rolling_std_30", "rolling_std_60",
        "volatility_7", "volatility_30", "momentum_7", "momentum_14",
        "arrival_change_7", "price_change_1", "temp_anomaly", "rain_anomaly",
        "price_x_arrivals", "roll_mean_x_volatility", "rainfall_x_arrivals", "temp_x_price",
        "high_price_regime", "high_arrival_regime", "volatility_regime",
        "sudden_spike_flag", "arrival_shock_flag", "price_residual_30", "normalized_dev_30"
    ]

    training_ds = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx", target="target_price", group_ids=["Mandi", "Commodity"],
        min_encoder_length=max_encoder_length // 2, max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, max_prediction_length=max_prediction_length,
        static_categoricals=["Mandi", "Commodity"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["target_price"] + unknown_reals,
        target_normalizer=GroupNormalizer(groups=["Mandi", "Commodity"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(training_ds, df, predict=True, stop_randomization=True)
    val_dataloader = validation_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    # 3. Load Model
    print(f"🧠 Loading Heavy Model Checkpoint...")
    import torchmetrics
    _orig = torchmetrics.Metric._apply
    def _safe(self, fn, *args, **kwargs):
        self._device = torch.device("cpu")
        return torch.nn.Module._apply(self, fn, *args, **kwargs)
    torchmetrics.Metric._apply = _safe
    
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    torchmetrics.Metric._apply = _orig
    model.eval()

    # 4. Inference
    print("🔮 Running inference on 14-day holdout set...")
    with torch.no_grad():
        predictions = model.predict(val_dataloader, mode="prediction", return_x=True)
        # Predictions are [batch, 14]
        # x contains "encoder_target" and "decoder_target"
        y_true = predictions.x["decoder_target"].numpy()
        y_pred = predictions.output.numpy()

    # 5. Metrics
    print("\n" + "-"*30)
    print("📈 FINAL EVALUATION METRICS")
    print("-"*30)
    
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    
    # Filter out any zero actuals to avoid infinite MAPE
    mask = flat_true > 0
    flat_true = flat_true[mask]
    flat_pred = flat_pred[mask]
    
    metrics = calculate_metrics(flat_true, flat_pred)
    
    for k, v in metrics.items():
        unit = "₹/qtl" if k == "MAE" else "%" if k in ["MAPE", "SMAPE", "Accuracy"] else ""
        print(f"{k:<10}: {v:>8.2f} {unit}")

    # Naive Baseline (Last Value)
    last_encoder_vals = predictions.x["encoder_target"][:, -1].unsqueeze(1).repeat(1, 14).numpy().flatten()[mask]
    naive_metrics = calculate_metrics(flat_true, last_encoder_vals)
    print(f"\nBaseline (Naive) Accuracy: {naive_metrics['Accuracy']:.2f}%")
    print(f"Model Improvement: {metrics['Accuracy'] - naive_metrics['Accuracy']:.2f} percentage points")
    print("="*60)

    # 6. Parity Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(flat_true, flat_pred, alpha=0.1, color='#3b82f6', s=1)
    plt.plot([flat_true.min(), flat_true.max()], [flat_true.min(), flat_true.max()], 'r--', lw=2)
    plt.title("TFT Heavy: Actual vs Predicted Prices")
    plt.xlabel("Actual Price (₹/qtl)")
    plt.ylabel("Predicted Price (₹/qtl)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TFT_Heavy_Parity_Plot.png")
    print("🖼️ Parity plot saved as 'TFT_Heavy_Parity_Plot.png'")

if __name__ == "__main__":
    main()
