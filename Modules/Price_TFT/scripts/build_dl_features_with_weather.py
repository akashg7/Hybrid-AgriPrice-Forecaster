import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

CROP_DATA = BASE_DIR / "DataScraping/CropData/ALL_CROPS_DATA.csv"
DISTRICT_MAP = BASE_DIR / "MergingFiles/full_with_district.csv"
WEATHER_DIR = BASE_DIR / "weather_data" 
OUTPUT_FILE = BASE_DIR / "data/processed/dl_advanced_features_data.csv"

def fourier_features(df, col, max_val):
    df['sin1'] = np.sin(2 * np.pi * df[col]/max_val)
    df['cos1'] = np.cos(2 * np.pi * df[col]/max_val)
    df['sin2'] = np.sin(4 * np.pi * df[col]/max_val)
    df['cos2'] = np.cos(4 * np.pi * df[col]/max_val)
    return df

def remove_outliers_iqr(group, col="ModalPrice"):
    # Standard solution for price data: Log-Transform before IQR.
    # Prices are log-normally distributed. This prevents negative lower bounds.
    log_vals = np.log1p(group[col])
    Q1 = log_vals.quantile(0.25)
    Q3 = log_vals.quantile(0.75)
    IQR = Q3 - Q1
    
    # Bounds in log space
    lower_bound_log = Q1 - 1.5 * IQR
    upper_bound_log = Q3 + 1.5 * IQR
    
    # Convert bounds back to normal space
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    
    group[col] = group[col].clip(lower=lower_bound, upper=upper_bound)
    return group

def process_group(g):
    g = g.sort_values("date").copy()
    g = remove_outliers_iqr(g, "ModalPrice")
    if "Arrivals" in g.columns:
        g = remove_outliers_iqr(g, "Arrivals")
    
    # --- 1. Target Definition (NO SHIFTING FEATURES, ONLY TARGET) ---
    g["target_price"] = g["ModalPrice"].shift(-1)
    
    # --- 2. Long & Short Term Lags ---
    for lag in [1, 3, 7, 14, 30, 60]:
        g[f"modal_lag_{lag}"] = g["ModalPrice"].shift(lag)
        g[f"arrivals_lag_{lag}"] = g["Arrivals"].shift(lag)
        
    # --- 3. Rolling Stats & Long-Term Memory ---
    for window in [7, 14, 30, 60]:
        g[f"rolling_mean_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).mean()
        g[f"rolling_std_{window}"] = g["ModalPrice"].rolling(window, min_periods=1).std().fillna(0)
        
        # Stability / CV feature
        g[f"volatility_{window}"] = (g[f"rolling_std_{window}"] / g[f"rolling_mean_{window}"]).replace([np.inf, -np.inf], 0).fillna(0)
        
        if window in [7, 30, 60]:
            g[f"arrivals_avg_{window}"] = g["Arrivals"].rolling(window, min_periods=1).mean()

    # Base Derived Stats
    g["price_range"] = g["MaxPrice"] - g["MinPrice"]
    g["price_spread"] = np.where(g["price_range"] > 0, (g["ModalPrice"] - g["MinPrice"]) / g["price_range"], 0)
    
    for lag in [7, 14]:
        g[f"momentum_{lag}"] = (g["ModalPrice"] / g[f"modal_lag_{lag}"]).replace([np.inf, -np.inf], 0).fillna(1)
        
    g["arrival_change_7"] = (g["Arrivals"] / g["arrivals_lag_7"]).replace([np.inf, -np.inf], 0).fillna(1)
    
    # --- 4. Cross-Feature Interactions ---
    g["price_x_arrivals"] = g["ModalPrice"] * g["Arrivals"]
    g["roll_mean_x_volatility"] = g["rolling_mean_7"] * g["volatility_7"]
    
    if "rainfall" in g.columns:
        g["rainfall_x_arrivals"] = g["rainfall"] * g["Arrivals"]
        rain_30 = g["rainfall"].rolling(30, min_periods=1).mean()
        g["rain_anomaly"] = g["rainfall"] - rain_30
    else:
        g["rainfall_x_arrivals"] = 0

    if "temp_avg" in g.columns:
        g["temp_x_price"] = g["temp_avg"] * g["ModalPrice"]
        temp_30 = g["temp_avg"].rolling(30, min_periods=1).mean()
        g["temp_anomaly"] = g["temp_avg"] - temp_30
    else:
        g["temp_x_price"] = 0

    # --- 5. Regime / State Features ---
    g["high_price_regime"] = (g["ModalPrice"] > g["rolling_mean_30"]).astype(float)
    g["high_arrival_regime"] = (g["Arrivals"] > g["arrivals_avg_30"]).astype(float)
    
    mean_vol_7 = g["volatility_7"].rolling(30, min_periods=1).mean()
    g["volatility_regime"] = (g["volatility_7"] > mean_vol_7).astype(float)
    
    # --- 6. Spike Detection Features ---
    g["price_change_1"] = (g["ModalPrice"] / g["modal_lag_1"]).replace([np.inf, -np.inf], 0).fillna(1)
    g["sudden_spike_flag"] = (g["price_change_1"] > 1.1).astype(float)
    g["arrival_shock_flag"] = (g["arrival_change_7"] > 1.5).astype(float) # e.g. 50% spike
    
    # --- 7. Trend Decomposition & Stability ---
    # Residual from 30-day moving average as proxy for short-term noise / detrended
    g["price_residual_30"] = g["ModalPrice"] - g["rolling_mean_30"]
    # Normalized deviation (Z-score over 30 days)
    g["normalized_dev_30"] = np.where(g["rolling_std_30"] > 0, g["price_residual_30"] / g["rolling_std_30"], 0)

    return g

def main():
    print("=== Advanced Deep Learning Feature Builder ===")
    
    df = pd.read_csv(CROP_DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=["ModalPrice", "Arrivals"])
    
    # --- DOMAIN KNOWLEDGE ANOMALY REMOVAL ---
    # Prices < 100 Rs/Quintal (1 Rs/kg) are completely unrealistic for these 10 crops
    df = df[df["ModalPrice"] >= 100.0]
    
    # Arrivals < 0.05 Tonnes (50 kg) are likely data entry errors or micro-transactions
    df = df[df["Arrivals"] >= 0.05]
    
    df_map = pd.read_csv(DISTRICT_MAP, usecols=["Mandi", "district_name"]).drop_duplicates()
    df = pd.merge(df, df_map, on="Mandi", how="left")
    df['district_name'] = df['district_name'].fillna(df['Mandi'])
    
    weather_dfs = []
    weather_files = list(WEATHER_DIR.glob("weather_*.csv"))
    for f in weather_files:
        try:
            w_df = pd.read_csv(f)
            w_df["date"] = pd.to_datetime(w_df["date"]).dt.strftime('%Y-%m-%d')
            weather_dfs.append(w_df)
        except: pass
            
    if len(weather_dfs) > 0:
        all_weather = pd.concat(weather_dfs, ignore_index=True)
        all_weather = all_weather.drop_duplicates(subset=["district", "date"])
        df = pd.merge(df, all_weather, left_on=["district_name", "date"], right_on=["district", "date"], how="left")
    
    df["date"] = pd.to_datetime(df["date"])
    
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df = fourier_features(df, "day_of_year", 365.25)
    
    df = df.sort_values(["Mandi", "Commodity", "date"])
    df = df.groupby(["Mandi", "Commodity"], group_keys=False).apply(process_group)

    keep_columns = [
        "date", "Mandi", "Commodity", "ModalPrice", "target_price", "Arrivals",
        "day_of_year", "day_of_week", "month", "year", "sin1", "cos1", "sin2", "cos2",
        "temp_avg", "temp_max", "temp_min", "rainfall", "humidity", "solar_radiation", "wind_speed",
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
    
    final_cols = [c for c in keep_columns if c in df.columns]
    df_final = df[final_cols].copy()
    
    # Dropping logic carefully: keeping valid rows
    # Because of lag 60, we MUST drop where lag 60 is NaN, otherwise filling with 0 destroys time-series logic
    df_final = df_final.dropna(subset=["target_price", "modal_lag_60"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Built Adv Features: {df_final.shape}")
    print(f"✅ Saved directly to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
