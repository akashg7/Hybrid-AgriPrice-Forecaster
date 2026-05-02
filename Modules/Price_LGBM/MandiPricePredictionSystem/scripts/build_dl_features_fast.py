import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("DataScraping/CropData/ALL_CROPS_DATA.csv")
OUTPUT_FILE = Path("data/processed/dl_30_features_data.csv")

def fourier_features(df, col, max_val):
    df['sin1'] = np.sin(2 * np.pi * df[col]/max_val)
    df['cos1'] = np.cos(2 * np.pi * df[col]/max_val)
    df['sin2'] = np.sin(4 * np.pi * df[col]/max_val)
    df['cos2'] = np.cos(4 * np.pi * df[col]/max_val)
    return df

def process_group(g):
    g = g.sort_values("date").copy()
    
    # Target
    g["target_price"] = g["ModalPrice"].shift(-1)
    
    # Lags
    for lag in [1, 3, 7, 14]:
        g[f"modal_lag_{lag}"] = g["ModalPrice"].shift(lag)
        g[f"arrivals_lag_{lag}"] = g["Arrivals"].shift(lag)
        
    # Rolling Stats
    for window in [7, 14, 30]:
        g[f"rolling_mean_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).mean()
        g[f"rolling_std_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).std().fillna(0)
        
        # Volatility
        g[f"volatility_{window}"] = (g[f"rolling_std_{window}"] / g[f"rolling_mean_{window}"]).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Arrival Avgs
        if window in [7, 30]:
            g[f"arrivals_avg_{window}"] = g["Arrivals"].rolling(window, min_periods=1).mean()

    # Derived Price Stats
    g["price_range"] = g["MaxPrice"] - g["MinPrice"]
    g["price_spread"] = np.where(g["price_range"] > 0, (g["ModalPrice"] - g["MinPrice"]) / g["price_range"], 0)
    
    # Momentum
    for lag in [7, 14]:
        g[f"momentum_{lag}"] = (g["ModalPrice"] / g[f"modal_lag_{lag}"]).replace([np.inf, -np.inf], 0).fillna(1)
        
    # Arrival Shock
    g["arrival_change_7"] = (g["Arrivals"] / g["arrivals_lag_7"]).replace([np.inf, -np.inf], 0).fillna(1)

    return g

def main():
    print(f"Loading raw data from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    
    # Essential Date features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df = fourier_features(df, "day_of_year", 365.25)
    
    # Drop rows without prices or arrivals
    df = df.dropna(subset=["ModalPrice", "Arrivals", "MinPrice", "MaxPrice"])
    
    # Sort and Group by Mandi and Commodity
    print("Generating rolling lag features (ignoring missing weather)...")
    df = df.sort_values(["Mandi", "Commodity", "date"])
    df = df.groupby(["Mandi", "Commodity"], group_keys=False).apply(process_group)

    # Filter to only the 30 needed features for N-BEATS 
    keep_columns = [
        "date", "Mandi", "Commodity", "ModalPrice", "target_price", "Arrivals",
        "day_of_year", "day_of_week", "month", "year",
        "sin1", "cos1", "sin2", "cos2",
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
        "rolling_mean_30", "rolling_std_30", "price_range", "price_spread", 
        "volatility_7", "momentum_7", "momentum_14",
        "arrivals_lag_1", "arrivals_lag_7", "arrivals_avg_7", "arrival_change_7"
    ]
    
    df_final = df[keep_columns].copy()
    
    # Drop initial rows that have NaN targets/lags
    df_final = df_final.dropna(subset=["target_price", "modal_lag_14"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Generated 30-feature dataset with shape {df_final.shape}")
    print(f"✅ Saved directly to {OUTPUT_FILE}")
    print(f"✅ Earliest Date: {df_final['date'].min().date()} | Latest Date: {df_final['date'].max().date()}")

if __name__ == "__main__":
    main()
