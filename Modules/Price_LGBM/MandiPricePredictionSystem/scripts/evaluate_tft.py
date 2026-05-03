"""
evaluate_tft.py — Load trained TFT checkpoint and compute real metrics.
NO retraining. Just inference on the held-out test set.
"""

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
DATA_PATH = Path("data/processed/dl_30_features_data.csv")
CKPT_PATH = Path("epoch=9-step=43310.ckpt")
OUT_DIR = Path("outputs/eval")

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def main():
    print("=" * 60)
    print("  TFT Evaluation — Inference Only (No Retraining)")
    print("=" * 60)

    # 1. Load data (same as training script)
    print(f"\n1. Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Mandi", "Commodity", "date"])

    # Create time index per group (same as training)
    df["time_idx"] = df.groupby(["Mandi", "Commodity"]).cumcount()
    df["target_price"] = df["target_price"].clip(lower=1.0)
    df["group_id"] = df["Mandi"].astype(str) + "_" + df["Commodity"].astype(str)

    # Drop rows with missing required columns
    required_cols = ["target_price", "temp_avg", "humidity", "rainfall",
                     "rolling_mean_7", "volatility_7", "momentum_7",
                     "day_of_year", "sin1", "cos1"]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Fill any remaining NaNs in feature columns
    for col in ["temp_avg", "humidity", "rainfall", "rolling_mean_7", 
                "volatility_7", "momentum_7"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['date'].min().date()} — {df['date'].max().date()}")
    print(f"   Unique groups: {df['group_id'].nunique()}")

    max_prediction_length = 14
    max_encoder_length = 30
    training_cutoff = df["time_idx"].max() - max_prediction_length

    print(f"   Training cutoff time_idx: {training_cutoff}")
    print(f"   Max time_idx: {df['time_idx'].max()}")

    # 2. Rebuild EXACT same TimeSeriesDataSet as training
    print("\n2. Rebuilding TimeSeriesDataSet (exact same config as training)...")
    
    # Check which columns actually exist
    time_varying_known = ["time_idx"]
    for col in ["day_of_year", "sin1", "cos1"]:
        if col in df.columns:
            time_varying_known.append(col)
    
    time_varying_unknown = ["target_price"]
    for col in ["temp_avg", "humidity", "rainfall", "rolling_mean_7", 
                "volatility_7", "momentum_7"]:
        if col in df.columns:
            time_varying_unknown.append(col)
    
    print(f"   Known reals: {time_varying_known}")
    print(f"   Unknown reals: {time_varying_unknown}")

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
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    print(f"   Validation samples: {len(validation)}")

    # 3. Load TFT from checkpoint (NO training)
    print(f"\n3. Loading TFT from checkpoint: {CKPT_PATH}")
    
    # The checkpoint was trained on CUDA GPU. torchmetrics.Metric._apply()
    # tries to create a dummy tensor on self.device (cuda) before moving to
    # the target device, which crashes on Mac without CUDA.
    # Fix: monkey-patch torchmetrics to skip the problematic _apply.
    import torchmetrics
    
    _original_apply = torchmetrics.Metric._apply
    
    def _safe_apply(self, fn, *args, **kwargs):
        """Patched _apply that forces device to CPU before applying fn."""
        self._device = torch.device("cpu")
        return torch.nn.Module._apply(self, fn)
    
    torchmetrics.Metric._apply = _safe_apply
    
    try:
        raw_ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
        
        # Log checkpoint metadata
        if "epoch" in raw_ckpt:
            print(f"   Checkpoint epoch: {raw_ckpt['epoch']}")
        if "global_step" in raw_ckpt:
            print(f"   Global step: {raw_ckpt['global_step']}")
        if "hyper_parameters" in raw_ckpt:
            hp = raw_ckpt["hyper_parameters"]
            print(f"   hidden_size: {hp.get('hidden_size')}")
            print(f"   attention_head_size: {hp.get('attention_head_size')}")
            print(f"   dropout: {hp.get('dropout')}")
            print(f"   output_size: {hp.get('output_size')}")
            print(f"   learning_rate: {hp.get('learning_rate')}")
        
        # Remove callbacks that may hold CUDA references
        if "callbacks" in raw_ckpt:
            raw_ckpt["callbacks"] = {}
        
        # Save patched checkpoint
        import os
        tmp_ckpt = str(CKPT_PATH) + ".cpu_tmp.ckpt"
        torch.save(raw_ckpt, tmp_ckpt)
        
        best_model = TemporalFusionTransformer.load_from_checkpoint(
            tmp_ckpt, map_location="cpu"
        )
    finally:
        torchmetrics.Metric._apply = _original_apply
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
    
    best_model.eval()
    print("   ✓ Model loaded successfully (inference mode)")

    # 4. Run predictions
    print("\n4. Running inference on validation set...")
    predictions = best_model.predict(
        val_dataloader, 
        mode="prediction",  # returns point predictions (median quantile)
        return_x=True
    )

    # Get raw predictions (point forecasts from median quantile)
    raw_preds = best_model.predict(val_dataloader, mode="raw")

    # 5. Compute actuals vs predictions
    print("\n5. Computing metrics...")
    
    actuals_list = []
    preds_list = []
    
    for batch_idx, (x, y) in enumerate(val_dataloader):
        actuals_list.append(y[0])  # y is (target, weight) tuple
    
    actuals = torch.cat(actuals_list, dim=0).numpy()  # shape: (N, 14)
    
    if isinstance(predictions, tuple):
        preds = predictions[0].numpy()
    else:
        preds = predictions.numpy()

    # Flatten for global metrics
    y_true = actuals.flatten()
    y_pred = preds.flatten()

    # Remove any zero/negative actuals for cleaner metrics
    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n_samples = len(actuals)
    n_points = len(y_true)

    # Naive baseline: last encoder value repeated
    naive_list = []
    for batch_idx, (x, y) in enumerate(val_dataloader):
        encoder_target = x["encoder_target"]  # (batch, encoder_len)
        last_val = encoder_target[:, -1].unsqueeze(1).expand(-1, max_prediction_length)
        naive_list.append(last_val)
    naive_all = torch.cat(naive_list, dim=0).numpy().flatten()
    naive_all = naive_all[mask]

    # Compute all metrics
    tft_mae = mae(y_true, y_pred)
    tft_rmse = rmse(y_true, y_pred)
    tft_mape = mape(y_true, y_pred)
    tft_smape = smape(y_true, y_pred)
    tft_acc = 100.0 - tft_smape

    naive_mae_val = mae(y_true, naive_all)
    naive_rmse_val = rmse(y_true, naive_all)
    naive_mape_val = mape(y_true, naive_all)
    naive_smape_val = smape(y_true, naive_all)
    naive_acc = 100.0 - naive_smape_val

    print("\n" + "=" * 60)
    print("  ACTUAL TFT EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Validation samples: {n_samples}")
    print(f"  Total prediction points: {n_points}")
    print(f"  Prediction horizon: {max_prediction_length} days")
    
    print(f"\n  --- Naive Baseline (last-value repeat) ---")
    print(f"  MAE   : {naive_mae_val:.2f} ₹/quintal")
    print(f"  RMSE  : {naive_rmse_val:.2f}")
    print(f"  MAPE  : {naive_mape_val:.2f}%")
    print(f"  SMAPE : {naive_smape_val:.2f}%  (Accuracy ≈ {naive_acc:.2f}%)")
    
    print(f"\n  --- TFT (from checkpoint) ---")
    print(f"  MAE   : {tft_mae:.2f} ₹/quintal")
    print(f"  RMSE  : {tft_rmse:.2f}")
    print(f"  MAPE  : {tft_mape:.2f}%")
    print(f"  SMAPE : {tft_smape:.2f}%  (Accuracy ≈ {tft_acc:.2f}%)")
    
    print(f"\n  Improvement over naive:")
    print(f"  MAE reduction  : {naive_mae_val - tft_mae:.2f} ₹/quintal ({(1 - tft_mae/naive_mae_val)*100:.1f}%)")
    print(f"  SMAPE reduction: {naive_smape_val - tft_smape:.2f} pp")
    print("=" * 60)

    # 6. Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "model": ["naive_tft_eval", "tft_checkpoint"],
        "mae": [naive_mae_val, tft_mae],
        "rmse": [naive_rmse_val, tft_rmse],
        "mape": [naive_mape_val, tft_mape],
        "smape": [naive_smape_val, tft_smape],
        "accuracy_pct": [naive_acc, tft_acc],
        "n_val_samples": [n_samples, n_samples],
        "n_prediction_points": [n_points, n_points],
    })
    out_path = OUT_DIR / "eval_tft_checkpoint_actual.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Saved REAL metrics to: {out_path}")

if __name__ == "__main__":
    main()
