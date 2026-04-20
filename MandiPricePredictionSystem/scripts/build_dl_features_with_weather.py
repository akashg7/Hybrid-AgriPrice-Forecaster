import pandas as pd
import numpy as np
from pathlib import Path

# Paths
CROP_DATA = Path("DataScraping/CropData/ALL_CROPS_DATA.csv")
DISTRICT_MAP = Path("MergingFiles/full_with_district.csv")
WEATHER_DIR = Path("weather_data") # weather_collector default outputs here relative to CWD
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
    
    # Price and Arrival Lags
    for lag in [1, 3, 7, 14]:
        g[f"modal_lag_{lag}"] = g["ModalPrice"].shift(lag)
        g[f"arrivals_lag_{lag}"] = g["Arrivals"].shift(lag)
        
    # Rolling Stats
    for window in [7, 14, 30]:
        g[f"rolling_mean_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).mean()
        g[f"rolling_std_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).std().fillna(0)
        
        g[f"volatility_{window}"] = (g[f"rolling_std_{window}"] / g[f"rolling_mean_{window}"]).replace([np.inf, -np.inf], 0).fillna(0)
        
        if window in [7, 30]:
            g[f"arrivals_avg_{window}"] = g["Arrivals"].rolling(window, min_periods=1).mean()

    # Derived Stats
    g["price_range"] = g["MaxPrice"] - g["MinPrice"]
    g["price_spread"] = np.where(g["price_range"] > 0, (g["ModalPrice"] - g["MinPrice"]) / g["price_range"], 0)
    
    for lag in [7, 14]:
        g[f"momentum_{lag}"] = (g["ModalPrice"] / g[f"modal_lag_{lag}"]).replace([np.inf, -np.inf], 0).fillna(1)
        
    g["arrival_change_7"] = (g["Arrivals"] / g["arrivals_lag_7"]).replace([np.inf, -np.inf], 0).fillna(1)
    
    # Weather Anomalies
    if "temp_avg" in g.columns:
        temp_30 = g["temp_avg"].rolling(30, min_periods=1).mean()
        g["temp_anomaly"] = g["temp_avg"] - temp_30
        
    if "rainfall" in g.columns:
        rain_30 = g["rainfall"].rolling(30, min_periods=1).mean()
        g["rain_anomaly"] = g["rainfall"] - rain_30

    return g

def main():
    print("=== Deep Learning Native Feature Builder (With Local Weather API) ===")
    
    print(f"1. Loading Crop data: {CROP_DATA}")
    df = pd.read_csv(CROP_DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=["ModalPrice", "Arrivals"])
    
    print(f"2. Mapping Mandis to Districts: {DISTRICT_MAP}")
    # Extract existing mappings cleanly to circumvent missing master data
    df_map = pd.read_csv(DISTRICT_MAP, usecols=["Mandi", "district_name"]).drop_duplicates()
    df = pd.merge(df, df_map, on="Mandi", how="left")
    df['district_name'] = df['district_name'].fillna(df['Mandi']) # Fallback
    
    print("3. Loading Weather Data from API dumps...")
    weather_dfs = []
    weather_files = list(WEATHER_DIR.glob("weather_*.csv"))
    for f in weather_files:
        try:
            w_df = pd.read_csv(f)
            w_df["date"] = pd.to_datetime(w_df["date"]).dt.strftime('%Y-%m-%d')
            weather_dfs.append(w_df)
        except Exception as e:
             pass
            
    if len(weather_dfs) == 0:
        print("ERROR: Weather files not ready. Wait for the weather collector to finish!")
        return
        
    all_weather = pd.concat(weather_dfs, ignore_index=True)
    all_weather = all_weather.drop_duplicates(subset=["district", "date"])

    print("4. Merging Crop and Weather data...")
    # 'district' column in weather matches 'district_name' mapped fallback 
    df = pd.merge(df, all_weather, left_on=["district_name", "date"], right_on=["district", "date"], how="left")
    
    # Clean and parse merged date strings to datetime for operations
    df["date"] = pd.to_datetime(df["date"])
    
    # Essential Date Fourier Encoding
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df = fourier_features(df, "day_of_year", 365.25)
    
    print("5. Generating rolling spatial-temporal features per (Mandi, Commodity)...")
    df = df.sort_values(["Mandi", "Commodity", "date"])
    df = df.groupby(["Mandi", "Commodity"], group_keys=False).apply(process_group)

    print("6. Filtering down to optimal ~30 features for N-BEATS/Chronos...")
    keep_columns = [
        "date", "Mandi", "Commodity", "ModalPrice", "target_price", "Arrivals",
        "day_of_year", "day_of_week", "month", "year", "sin1", "cos1", "sin2", "cos2",
        "temp_avg", "temp_max", "temp_min", "rainfall", "humidity", "solar_radiation", "wind_speed",
        "modal_lag_1", "modal_lag_3", "modal_lag_7", "modal_lag_14",
        "rolling_mean_7", "rolling_std_7", "price_range", "volatility_7", "momentum_7",
        "arrivals_lag_1", "arrivals_lag_7", "arrival_change_7",
        "temp_anomaly", "rain_anomaly"
    ]
    
    # Drop rows without targets or missing too many lags natively
    final_cols = [c for c in keep_columns if c in df.columns]
    df_final = df[final_cols].copy()
    df_final = df_final.dropna(subset=["target_price", "modal_lag_14", "temp_avg"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Generated full weather-infused deep learning dataset: {df_final.shape}")
    print(f"✅ Saved directly to {OUTPUT_FILE}")
    print(f"✅ Extent: {df_final['date'].min().date()} — {df_final['date'].max().date()}")

if __name__ == "__main__":
    main()
