import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, SMAPE, RMSE
import torch

def train_multivariate():
    print("=== Advanced Deep Learning Pipeline (TFT Edition) ===")
    
    # 1. Load the advanced feature dataset
    print("Loading prepared dataset...")
    # Adjusting path to match typical execution from project root
    try:
        df = pd.read_csv("data/processed/dl_advanced_features_data.csv")
    except:
        df = pd.read_csv("../data/processed/dl_advanced_features_data.csv") # Fallback
    
    # Add time index required by PyTorch Forecasting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Mandi", "Commodity", "date"])
    
    # Create purely sequential integer time index per group
    df["time_idx"] = df.groupby(["Mandi", "Commodity"]).cumcount()
    
    # Ensure all targets are positive for relative scaling
    df["target_price"] = df["target_price"].clip(lower=1.0)
    
    # Create string format for groupings
    df["Mandi"] = df["Mandi"].astype(str)
    df["Commodity"] = df["Commodity"].astype(str)

    max_prediction_length = 14 # Predict next 14 days
    max_encoder_length = 60    # Elevated context lookback (60 days)
    
    # Time-based exact split (NO random)
    training_cutoff = df["time_idx"].max() - max_prediction_length

    print("Configuring TimeSeriesDataSet...")
    
    # Unknown reals that depend strictly on historic values
    unknown_reals = [
        "Arrivals", "temp_avg", "temp_max", "temp_min", "rainfall", "humidity", "solar_radiation", "wind_speed",
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14", "modal_lag_30", "modal_lag_60",
        "arrivals_lag_1", "arrivals_lag_7", "arrivals_lag_30", "arrivals_lag_60",
        "rolling_mean_7", "rolling_mean_30", "rolling_mean_60",
        "rolling_std_7", "rolling_std_30", "rolling_std_60",
        "volatility_7", "volatility_30", "momentum_7", "momentum_14",
        "arrival_change_7", "price_change_1",
        "temp_anomaly", "rain_anomaly",
        "price_x_arrivals", "roll_mean_x_volatility", "rainfall_x_arrivals", "temp_x_price",
        "high_price_regime", "high_arrival_regime", "volatility_regime",
        "sudden_spike_flag", "arrival_shock_flag",
        "price_residual_30", "normalized_dev_30"
    ]
    # Filter to only columns that actually exist to avoid crash if some are all-NaN/dropped
    unknown_reals = [f for f in unknown_reals if f in df.columns]

    # Define exact TimeSeries boundaries
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_price",
        group_ids=["Mandi", "Commodity"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Mandi", "Commodity"],
        time_varying_known_reals=["time_idx", "day_of_year", "day_of_week", "month", "sin1", "cos1", "sin2", "cos2"],
        time_varying_unknown_reals=["target_price"] + unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["Mandi", "Commodity"], transformation="softplus"
        ),  
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 2. Creating dataloaders for the GPU/MPS
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    batch_size = 128  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=4)

    # 3. Architect TFT (Temporal Fusion Transformer)
    print("Instantiating Multi-variate TFT Model for exact external shock modeling...")
    pl.seed_everything(42)
    net = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,             # Tuned learning rate
        hidden_size=64,                 # Strengthened hidden representation
        attention_head_size=4,          # Expanded attention mapping 
        dropout=0.2,                    # Prevent Overfitting
        hidden_continuous_size=16,
        output_size=7,                  # Quantiles for confidence bands!
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 4. Spin up the Lightning Trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode="min"
    )
    lr_logger = LearningRateMonitor()  # Logs the learning rate
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-advanced-tft-{epoch:02d}-{val_loss:.2f}"
    )
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto", # auto detects MPS/GPU/CPU
        devices=1,
        gradient_clip_val=0.1, # Gradient clipping
        callbacks=[early_stop_callback, checkpoint_callback, lr_logger],
    )

    # 5. Execute Training!
    print(f"Executing High-Performance Multi-variate Training...")
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Optimization Complete! Best multi-variate model saved.")
    
    # --- INTERPRETABILITY TOOLS ---
    print("Generating Global Feature Importance and Attention...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    else:
        best_tft = net

    # Run evaluation across sample
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    
    # 1. Feature Importance plotting
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    figs = best_tft.plot_interpretation(interpretation)
    for key, fig in figs.items():
        fig.savefig(f"tft_feature_importance_{key}.png")
    
    print("✅ Feature importance successfully mapped out to PNGs. Model represents State-of-the-Art config setup.")

if __name__ == "__main__":
    train_multivariate()
