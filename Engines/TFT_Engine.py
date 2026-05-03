import torch
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings("ignore")

class TFTEngine:
    def __init__(self, model_path, features_path):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.df = None
        self.model = None
        self.training_ds = None
        self.load_engine()

    def load_engine(self):
        print("🤖 Loading TFT Heavy Engine...")
        self.df = pd.read_csv(self.features_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values(["Mandi", "Commodity", "date"])
        self.df["time_idx"] = self.df.groupby(["Mandi", "Commodity"]).cumcount()
        
        # Stability filter
        counts = self.df.groupby(["Mandi", "Commodity"]).size()
        self.df = self.df.set_index(["Mandi", "Commodity"]).loc[counts[counts >= 60].index].reset_index()
        
        # Clean numeric data
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df.groupby(["Mandi", "Commodity"])[numeric_cols].ffill().fillna(0)

        # CPU Patch
        import torchmetrics
        _orig = torchmetrics.Metric._apply
        def _safe_apply(self, fn, *args, **kwargs):
            self._device = torch.device("cpu")
            return torch.nn.Module._apply(self, fn, *args, **kwargs)
        torchmetrics.Metric._apply = _safe_apply
        
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path, map_location="cpu").eval()
        torchmetrics.Metric._apply = _orig

        # Dataset Definition (52 features)
        known_reals = ['time_idx', 'day_of_year', 'day_of_week', 'month', 'sin1', 'cos1', 'sin2', 'cos2']
        unknown_reals = ['target_price', 'Arrivals', 'temp_avg', 'temp_max', 'temp_min', 'rainfall', 'humidity', 'solar_radiation', 'wind_speed', 'modal_lag_1', 'modal_lag_3', 'modal_lag_7', 'modal_lag_14', 'modal_lag_30', 'modal_lag_60', 'arrivals_lag_1', 'arrivals_lag_7', 'arrivals_lag_30', 'arrivals_lag_60', 'rolling_mean_7', 'rolling_mean_30', 'rolling_mean_60', 'rolling_std_7', 'rolling_std_30', 'rolling_std_60', 'volatility_7', 'volatility_30', 'momentum_7', 'momentum_14', 'arrival_change_7', 'price_change_1', 'temp_anomaly', 'rain_anomaly', 'price_x_arrivals', 'roll_mean_x_volatility', 'rainfall_x_arrivals', 'temp_x_price', 'high_price_regime', 'high_arrival_regime', 'volatility_regime', 'sudden_spike_flag', 'arrival_shock_flag', 'price_residual_30', 'normalized_dev_30']

        self.training_ds = TimeSeriesDataSet(
            self.df[lambda x: x.time_idx <= self.df["time_idx"].max() - 30],
            time_idx="time_idx", target="target_price", group_ids=["Mandi", "Commodity"],
            max_encoder_length=60, max_prediction_length=14,
            static_categoricals=["Mandi", "Commodity"],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["Mandi", "Commodity"], transformation="softplus"),
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        )

    def predict(self, mandi, commodity):
        gdf = self.df[(self.df["Mandi"] == mandi) & (self.df["Commodity"] == commodity)].copy()
        
        # REQUIRED: 60 (encoder) + 14 (decoder) = 74 minimum rows
        if len(gdf) < 74: 
            print(f"⚠️ Insufficient data for {mandi}/{commodity}: {len(gdf)} rows found, 74 required.")
            return None
            
        hist, actual = gdf.iloc[:-14], gdf.tail(14)
        try:
            pred_ds = TimeSeriesDataSet.from_dataset(self.training_ds, hist, predict=True, stop_randomization=True)
        except Exception as e:
            print(f"❌ Dataset Generation Error: {e}")
            return None

        with torch.no_grad():
            out = self.model.predict(pred_ds.to_dataloader(train=False, batch_size=1), mode="raw")
            ts = out.output[0] if hasattr(out, "output") else out[0]
            p50 = ts[0, :, 3].cpu().numpy().tolist()
            p90 = ts[0, :, 5].cpu().numpy().tolist()
        
        return {
            "history": [[d.strftime('%Y-%m-%d'), p] for d, p in hist.tail(30)[["date", "ModalPrice"]].values.tolist()],
            "actual": [[d.strftime('%Y-%m-%d'), p] for d, p in actual[["date", "ModalPrice"]].values.tolist()],
            "forecast": [[(hist["date"].iloc[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'), round(float(p50[i]), 2)] for i in range(14)],
            "forecast_high": [[(hist["date"].iloc[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'), round(float(p90[i]), 2)] for i in range(14)],
            "momentum": hist.tail(30)["momentum_7"].values.tolist(),
            "shocks": hist.tail(30)["arrival_shock_flag"].values.tolist()
        }
