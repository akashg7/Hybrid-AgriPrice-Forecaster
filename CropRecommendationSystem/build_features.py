# CropRecommendationSystem/build_features.py
# ============================================================================
# Feature Engineering Pipeline for Crop & Fertilizer Recommendation
# ============================================================================
# Mirrors the depth of the price forecaster's build_training_data.py
# Expands raw 9-column data_core.csv into 40+ engineered features.
# ============================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

def load_raw_data() -> pd.DataFrame:
    """Load and normalize Kaggle Crop_recommendation.csv (from archive.zip)."""
    df = pd.read_csv(DATA_PATH)
    # Columns: N, P, K, temperature, humidity, ph, rainfall, label
    df.rename(columns={
        'N': 'nitrogen',
        'P': 'phosphorous',
        'K': 'potassium',
        'temperature': 'temperature',
        'humidity': 'humidity',
        'ph': 'ph',
        'rainfall': 'rainfall',
        'label': 'crop_type',
    }, inplace=True)
    return df


# ============================================================================
# 1. Nutrient Interaction Features
# ============================================================================
def add_nutrient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NPK ratios and interactions are critical in agronomy.
    Different crops need very specific N:P:K ratios.
    e.g., leafy crops need high N, root crops need high K.
    """
    eps = 1e-6  # prevent division by zero
    # Note: This dataset does NOT have a 'fertilizer' or 'soil_type' column.
    # It has: nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, crop_type

    # Total macro-nutrient load
    df['npk_total'] = df['nitrogen'] + df['phosphorous'] + df['potassium']

    # Individual nutrient ratios (fraction of total)
    df['n_ratio'] = df['nitrogen'] / (df['npk_total'] + eps)
    df['p_ratio'] = df['phosphorous'] / (df['npk_total'] + eps)
    df['k_ratio'] = df['potassium'] / (df['npk_total'] + eps)

    # Pairwise ratios (agronomically meaningful)
    df['n_to_p'] = df['nitrogen'] / (df['phosphorous'] + eps)
    df['n_to_k'] = df['nitrogen'] / (df['potassium'] + eps)
    df['p_to_k'] = df['phosphorous'] / (df['potassium'] + eps)

    # Interaction products (captures non-linear nutrient synergies)
    df['n_times_p'] = df['nitrogen'] * df['phosphorous']
    df['n_times_k'] = df['nitrogen'] * df['potassium']
    df['p_times_k'] = df['phosphorous'] * df['potassium']
    df['npk_product'] = df['nitrogen'] * df['phosphorous'] * df['potassium']

    # Nutrient dominance flag (which macro-nutrient dominates the soil)
    df['dominant_nutrient'] = np.where(
        (df['nitrogen'] >= df['phosphorous']) & (df['nitrogen'] >= df['potassium']), 0,  # N dominant
        np.where(
            (df['phosphorous'] >= df['nitrogen']) & (df['phosphorous'] >= df['potassium']), 1,  # P dominant
            2  # K dominant
        )
    )

    # Nutrient balance score (how evenly distributed are N, P, K)
    # Low variance = balanced; high variance = specialized
    df['nutrient_balance'] = df[['nitrogen', 'phosphorous', 'potassium']].std(axis=1)
    df['nutrient_cv'] = df['nutrient_balance'] / (df[['nitrogen', 'phosphorous', 'potassium']].mean(axis=1) + eps)

    return df


# ============================================================================
# 2. Climate Interaction Features
# ============================================================================
def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather interactions drive crop viability.
    e.g., High temp + low humidity = drought stress.
    pH interactions with rainfall affect nutrient availability.
    """
    eps = 1e-6

    # Heat-Humidity Index (simplified agricultural heat stress proxy)
    df['heat_index'] = df['temperature'] * df['humidity'] / 100.0

    # Rainfall intensity relative to humidity
    df['rain_humidity_ratio'] = df['rainfall'] / (df['humidity'] + eps)

    # Aridity proxy: high temp + low rainfall = arid conditions
    df['aridity_index'] = df['temperature'] / (df['rainfall'] + eps)

    # Climate-nutrient interactions
    # Warm + wet + high-N → ideal for leafy crops
    df['temp_x_rainfall'] = df['temperature'] * df['rainfall']
    df['temp_x_humidity'] = df['temperature'] * df['humidity']
    df['humidity_x_rainfall'] = df['humidity'] * df['rainfall']

    # pH interactions (critical for nutrient availability)
    df['ph_x_nitrogen'] = df['ph'] * df['nitrogen']
    df['ph_x_rainfall'] = df['ph'] * df['rainfall']
    df['ph_deviation_from_neutral'] = abs(df['ph'] - 7.0)  # deviation from neutral
    df['ph_squared'] = df['ph'] ** 2

    # Quadratic terms for non-linear climate sensitivity
    df['temp_squared'] = df['temperature'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    df['rainfall_squared'] = df['rainfall'] ** 2
    df['rainfall_log'] = np.log1p(df['rainfall'])  # log transform for skewed rainfall

    return df


# ============================================================================
# 3. Soil-Nutrient Climate Cross Features
# ============================================================================
def add_rainfall_bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rainfall regime is critical for crop selection.
    Binning rainfall into categories captures non-linear effects.
    """
    # Rainfall regime bins
    df['rainfall_regime'] = pd.cut(
        df['rainfall'],
        bins=[0, 50, 100, 150, 200, 300, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    # pH regime bins (acidic / neutral / alkaline)
    df['ph_regime'] = pd.cut(
        df['ph'],
        bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Temperature regime bins
    df['temp_regime'] = pd.cut(
        df['temperature'],
        bins=[0, 20, 25, 30, 35, 50],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    return df


# ============================================================================
# 4. Statistical Aggregate Features (per-crop and per-soil baselines)
# ============================================================================
def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How does this sample compare to the global average?
    This is analogous to the price forecaster's rolling_mean / zscore features.
    """
    # Global z-scores for key features
    for col in ['temperature', 'rainfall', 'nitrogen', 'phosphorous', 'potassium', 'ph', 'humidity']:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[f'{col}_zscore'] = (df[col] - col_mean) / (col_std + 1e-6)

    return df


# ============================================================================
# 5. Encode Categoricals
# ============================================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode the crop target."""
    from sklearn.preprocessing import LabelEncoder

    encoders = {}

    # Encode crop target
    le_crop = LabelEncoder()
    df['crop_type_encoded'] = le_crop.fit_transform(df['crop_type'])
    encoders['crop_type'] = le_crop

    return df, encoders


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("=" * 70)
    print("🌾 CROP RECOMMENDATION: FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    print(f"\n📂 Loading raw data from {DATA_PATH} ...")
    df = load_raw_data()
    print(f"   Raw shape: {df.shape} ({df.shape[1]} columns)")
    print(f"   Crops: {df['crop_type'].nunique()} unique → {sorted(df['crop_type'].unique())}")

    print("\n⚙️  Phase 1: Nutrient Interaction Features...")
    df = add_nutrient_features(df)

    print("⚙️  Phase 2: Climate & pH Interaction Features...")
    df = add_climate_features(df)

    print("⚙️  Phase 3: Rainfall & pH Regime Binning...")
    df = add_rainfall_bin_features(df)

    print("⚙️  Phase 4: Statistical Aggregate Features...")
    df = add_aggregate_features(df)

    print("⚙️  Phase 5: Encoding Categoricals...")
    df, encoders = encode_categoricals(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full engineered dataset
    parquet_path = OUT_DIR / "crop_features_engineered.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n✅ Saved engineered data: {parquet_path}")
    print(f"   Final shape: {df.shape} ({df.shape[1]} columns)")
    print(f"   Expanded from 9 → {df.shape[1]} features")

    # Save encoders
    import joblib
    encoder_path = OUT_DIR / "label_encoders.pkl"
    joblib.dump(encoders, encoder_path)
    print(f"   Saved label encoders: {encoder_path}")

    # Print feature list
    print(f"\n📊 Feature columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"   {i+1:3d}. {col}")


if __name__ == "__main__":
    main()
