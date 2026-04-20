import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch

def train_multivariate():
    print("=== Deep Learning Pipeline (Temporal Fusion Transformer Edition) ===")
    
    # 1. Load the exact 30-feature dataset we curated
    print("Loading prepared dataset...")
    df = pd.read_csv("dl_30_features_data.csv")
    
    # Add time index required by PyTorch Forecasting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Mandi", "Commodity", "date"])
    
    # Create purely sequential integer time index per group
    df["time_idx"] = df.groupby(["Mandi", "Commodity"]).cumcount()
    
    # Ensure all targets are positive for relative scaling
    df["target_price"] = df["target_price"].clip(lower=1.0)
    
    # For speed, map strings to categorical codes
    df["group_id"] = df["Mandi"].astype(str) + "_" + df["Commodity"].astype(str)

    max_prediction_length = 14 # Predict next 14 days
    max_encoder_length = 30    # Lookback window of 30 days
    training_cutoff = df["time_idx"].max() - max_prediction_length

    print("Configuring TimeSeriesDataSet...")
    
    # Define exact TimeSeries boundaries (TFT loves covariates!)
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_price",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Mandi", "Commodity"],
        time_varying_known_reals=["time_idx", "day_of_year", "sin1", "cos1"],
        time_varying_unknown_reals=[
            "target_price", "temp_avg", "humidity", "rainfall", 
            "rolling_mean_7", "volatility_7", "momentum_7"
        ],
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),  
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 2. Creating dataloaders for the GPU
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    batch_size = 128  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    # 3. Architect TFT (Temporal Fusion Transformer)
    print("Instantiating Multi-variate TFT Model for exact external shock modeling...")
    pl.seed_everything(42)
    net = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # Quantiles for confidence bands!
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 4. Spin up the Lightning Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-tft-model-{epoch:02d}-{val_loss:.2f}"
    )
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 5. Execute Training!
    print(f"Executing Multi-variate GPU Optimization...")
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Optimization Complete! Best multi-variate model saved.")

if __name__ == "__main__":
    train_multivariate()
