# scripts/prepare_moirai_data.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

RAW_PATH = Path("data/processed/mandi_feature_engineered.csv")
OUT_PATH = Path("data/processed/mandi_moirai_scaled.parquet")
SCALER_DIR = Path("models")
SCALER_DIR.mkdir(parents=True, exist_ok=True)

DYN_SCALER_PATH = SCALER_DIR / "moirai_dyn_scaler.pkl"
TARGET_SCALER_PATH = SCALER_DIR / "moirai_target_scaler.pkl"


def main():
    print(f"Loading engineered data from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)

    # ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---------- choose dynamic numeric cols (exclude IDs, series_id, target ----------
    dynamic_num_cols = [
        # core prices & arrivals
        "Arrivals", "MinPrice", "MaxPrice",
        # weather
        "temp_avg", "temp_max", "temp_min", "rainfall",
        "humidity", "solar_radiation", "wind_speed",
        # time encodings (optional but fine to scale)
        "day", "month", "year", "day_of_week", "week_of_year",
        "day_of_year", "is_weekend",
        "sin1", "cos1", "sin2", "cos2",
        # price lags & stats
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14", "modal_lag_30",
        "rolling_mean_3", "rolling_std_3",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_30", "rolling_std_30",
        "price_range", "price_spread", "norm_spread",
        "volatility_7", "volatility_30", "zscore_7",
        "momentum_7", "momentum_14",
        # arrivals features
        "arrivals_lag_1", "arrivals_lag_3", "arrivals_lag_7", "arrivals_lag_14",
        "arrivals_avg_7", "arrivals_avg_30", "arrival_change_7",
        # climate aggregates
        "temp_range", "rain_intensity", "temp_avg_30", "rain_avg_30",
        "temp_anomaly", "rain_anomaly",
        # geo encodings
        "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    ]

    # sanity: keep only those that actually exist
    dynamic_num_cols = [c for c in dynamic_num_cols if c in df.columns]

    print(f"Using {len(dynamic_num_cols)} dynamic numeric cols.")

    # ---------- time-based split: train / val / test ----------
    # 1) first split: train vs temp (80/20)
    train_df, temp_df = train_test_split(df, test_size=0.2, shuffle=False)
    # 2) split temp into val/test (50/50 -> 10% + 10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # ---------- scale dynamic numeric features ----------
    dyn_scaler = RobustScaler()
    dyn_scaler.fit(train_df[dynamic_num_cols])

    train_df.loc[:, dynamic_num_cols] = dyn_scaler.transform(train_df[dynamic_num_cols])
    val_df.loc[:, dynamic_num_cols] = dyn_scaler.transform(val_df[dynamic_num_cols])
    test_df.loc[:, dynamic_num_cols] = dyn_scaler.transform(test_df[dynamic_num_cols])

    # ---------- scale target (ModalPrice) into 'y' ----------
    target_scaler = RobustScaler()

    train_df["y"] = target_scaler.fit_transform(train_df[["ModalPrice"]])
    val_df["y"] = target_scaler.transform(val_df[["ModalPrice"]])
    test_df["y"] = target_scaler.transform(test_df[["ModalPrice"]])

    # ---------- recombine in original time order ----------
    scaled_df = pd.concat([train_df, val_df, test_df], axis=0)
    scaled_df = scaled_df.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scaled_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved scaled data to {OUT_PATH} with shape {scaled_df.shape}")

    # save scalers
    joblib.dump(dyn_scaler, DYN_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    print(f"Saved dyn_scaler to {DYN_SCALER_PATH}")
    print(f"Saved target_scaler to {TARGET_SCALER_PATH}")


if __name__ == "__main__":
    main()
