# scripts/build_training_data.py
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("merged_crop_data_with_weather.csv")
OUT_PATH = Path("data/processed/training_data.parquet")


def add_lag_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Mandi, Commodity) group:
    - add price lags (1,2,3,7 days)
    - add target = next day's ModalPrice
    """
    df = df.sort_values(["Mandi", "Commodity", "date"]).copy()

    def process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        for lag in [1, 2, 3, 7]:
            g[f"lag_{lag}"] = g["ModalPrice"].shift(lag)
        # target is next day's price
        g["target_price"] = g["ModalPrice"].shift(-1)
        return g

    df = df.groupby(["Mandi", "Commodity"], group_keys=False).apply(process_group)

    # remove rows where target or important lags are NaN (start/end of series)
    lag_cols = [f"lag_{l}" for l in [1, 2, 3, 7]]
    df = df.dropna(subset=lag_cols + ["target_price"])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features like day of week, month."""
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month
    return df


def main():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw data from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)

    # parse dates
    df["date"] = pd.to_datetime(df["date"])

    # keep only columns we need for now
    needed_cols = ["date", "Mandi", "Commodity", "ModalPrice"]
    df = df[needed_cols].copy()

    print("Adding lag and target columns...")
    df = add_lag_and_target(df)

    print("Adding time features...")
    df = add_time_features(df)

    # treat Mandi & Commodity as categories (LightGBM can handle this)
    df["Mandi"] = df["Mandi"].astype("category")
    df["Commodity"] = df["Commodity"].astype("category")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved training data with shape {df.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
