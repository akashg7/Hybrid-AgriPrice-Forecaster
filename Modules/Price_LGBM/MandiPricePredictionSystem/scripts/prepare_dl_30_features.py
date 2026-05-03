# scripts/prepare_dl_30_features.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Look for processed data (can be CSV or parquet)
DATA_PATH = Path("data/processed/mandi_feature_engineered.csv") 
if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/training_data.parquet")

OUT_PATH = Path("data/processed/dl_30_features_data.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Cannot find source data at {DATA_PATH} or alternative paths.")
        return

    print(f"Loading data from {DATA_PATH} ...")
    if DATA_PATH.suffix == '.csv':
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_parquet(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # 30 most crucial features for Deep Learning
    keep_columns = [
        "date", "Mandi", "Commodity", "ModalPrice", "Arrivals",
        "day", "month", "year", "day_of_week", "day_of_year",
        "sin1", "cos1",
        "temp_avg", "temp_max", "temp_min", "rainfall",
        "humidity", "solar_radiation", "wind_speed",
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14",
        "rolling_mean_7", "rolling_std_7",
        "price_range", "volatility_7", "momentum_7",
        "arrivals_lag_1", "arrivals_lag_7", "arrival_change_7",
        "temp_anomaly", "rain_anomaly",
        "lat_sin", "lat_cos", "lon_sin", "lon_cos"
    ]

    # Filter columns to only those that actually exist in the dataframe
    final_cols = [c for c in keep_columns if c in df.columns]
    
    print(f"Found {len(final_cols)} matching features. Dropping the rest...")
    df_reduced = df[final_cols].copy()
    
    # Sort and write
    df_reduced = df_reduced.sort_values(by=["Mandi", "Commodity", "date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_reduced.to_csv(OUT_PATH, index=False)
    
    print(f"\nFinal Shape: {df_reduced.shape}")
    print(f"Saved reduced deep learning dataset to: {OUT_PATH}\n")
    print(f"NOTE: The web scraper 'fast_scrape.py' has been patched to scrape up to {pd.Timestamp.today().date()}")
    print("Run the scraper and re-merge weather data to update to today's date.")


if __name__ == "__main__":
    main()
