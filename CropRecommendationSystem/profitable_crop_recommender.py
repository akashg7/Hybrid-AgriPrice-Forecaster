import os
import sys
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Paths for the Crop Recommendation system
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_lgbm.pkl")
CROP_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")
CROP_SCALER_PATH = os.path.join(MODEL_DIR, "crop_feature_scaler.pkl")
DATA_PARQUET_PATH = os.path.join(DATA_DIR, "crop_features_engineered.parquet")

# Path for the Price Forecasts (from the main pipeline)
FORECAST_CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "outputs", "forecasts_moirai", "mandi_commodity_all_forecasts_moirai.csv")
# Fallback if LightGBM used
FORECAST_CSV_PATH_LGBM = os.path.join(os.path.dirname(BASE_DIR), "outputs", "forecasts", "mandi_commodity_all_forecasts.csv")

def engineer_single_input(n, p, k, temp, humidity, ph, rainfall, df_ref):
    eps = 1e-6
    npk_total = n + p + k
    row = {
        'nitrogen': n, 'phosphorous': p, 'potassium': k,
        'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
        'npk_total': npk_total,
        'n_ratio': n / (npk_total + eps), 'p_ratio': p / (npk_total + eps), 'k_ratio': k / (npk_total + eps),
        'n_to_p': n / (p + eps), 'n_to_k': n / (k + eps), 'p_to_k': p / (k + eps),
        'n_times_p': n * p, 'n_times_k': n * k, 'p_times_k': p * k,
        'npk_product': n * p * k,
        'dominant_nutrient': 0 if (n >= p and n >= k) else (1 if (p >= n and p >= k) else 2),
        'nutrient_balance': np.std([n, p, k]),
        'nutrient_cv': np.std([n, p, k]) / (np.mean([n, p, k]) + eps),
        'heat_index': temp * humidity / 100.0,
        'rain_humidity_ratio': rainfall / (humidity + eps),
        'aridity_index': temp / (rainfall + eps),
        'temp_x_rainfall': temp * rainfall, 'temp_x_humidity': temp * humidity,
        'humidity_x_rainfall': humidity * rainfall,
        'ph_x_nitrogen': ph * n, 'ph_x_rainfall': ph * rainfall,
        'ph_deviation_from_neutral': abs(ph - 7.0), 'ph_squared': ph ** 2,
        'temp_squared': temp ** 2, 'humidity_squared': humidity ** 2,
        'rainfall_squared': rainfall ** 2, 'rainfall_log': np.log1p(rainfall),
        'rainfall_regime': 0 if rainfall<=50 else (1 if rainfall<=100 else (2 if rainfall<=150 else (3 if rainfall<=200 else (4 if rainfall<=300 else 5)))),
        'ph_regime': 0 if ph<=5.5 else (1 if ph<=6.5 else (2 if ph<=7.5 else (3 if ph<=8.5 else 4))),
        'temp_regime': 0 if temp<=20 else (1 if temp<=25 else (2 if temp<=30 else (3 if temp<=35 else 4))),
    }
    
    for col in ['temperature', 'rainfall', 'nitrogen', 'phosphorous', 'potassium', 'ph', 'humidity']:
        row[f'{col}_zscore'] = (row[col] - df_ref[col].mean()) / (df_ref[col].std() + eps)
        
    return pd.DataFrame([row])

def recommend_best_profitable_crop(mandi, n, p, k, temp, humidity, ph, rainfall):
    print("==========================================================")
    print("🌱 INTEGRATED CROP RECOMMENDATION & PRICE FORECASTER")
    print("==========================================================")
    
    if not os.path.exists(CROP_MODEL_PATH):
        print("❌ Error: Crop model not found. Please run 'train_crop_model.py' first.")
        return
        
    crop_model = joblib.load(CROP_MODEL_PATH)
    encoders = joblib.load(CROP_ENCODER_PATH)
    crop_le = encoders['crop_type']
    crop_scaler = joblib.load(CROP_SCALER_PATH)
    df_ref = pd.read_parquet(DATA_PARQUET_PATH)
    
    # 1. Feature Engineering (47 cols -> 45 features)
    input_df = engineer_single_input(n, p, k, temp, humidity, ph, rainfall, df_ref)
    train_features = [col for col in df_ref.columns if col not in ['crop_type', 'crop_type_encoded']]
    for col in train_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[train_features]
    
    try:
        input_scaled = crop_scaler.transform(input_df.values)
    except Exception as e:
        print(f"❌ Feature mismatch. {e}")
        return
    
    # 2. Predict Top Agronomic Crops
    probabilities = crop_model.predict_proba(input_scaled)[0]
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_crops = crop_le.inverse_transform(top_3_indices)
    top_3_probs = probabilities[top_3_indices]
    
    print(f"\n🌍 Location: {mandi}")
    print(f"🌡️ Weather Conditions -> Temp: {temp}°C | Rainfall: {rainfall}mm | Humidity: {humidity}%")
    print("\n✅ TOP 3 AGRONOMICALLY SUITABLE CROPS:")
    for crop, prob in zip(top_3_crops, top_3_probs):
        print(f"   - {crop.capitalize()} (Feasibility: {prob*100:.2f}%)")
        
    # 3. Integrate with the 14-Day Price Forecast Pipeline
    print("\n💰 EVALUATING MARKET PROFITABILITY (14-Day Forecast)...")
    
    target_csv = FORECAST_CSV_PATH if os.path.exists(FORECAST_CSV_PATH) else FORECAST_CSV_PATH_LGBM
    
    if not os.path.exists(target_csv):
        print(f"\n⚠️ Notice: No existing price forecasts found at {target_csv}.")
        print("💡 To see projected profitability, ensure you ran the Price Forecaster pipeline first.")
        print(f"\n🏆 Final Recommendation (Agronomic Only): Grow **{top_3_crops[0].title()}**")
        return
        
    df_forecasts = pd.read_csv(target_csv)
    best_crop = top_3_crops[0]
    highest_price = 0
    
    for crop in top_3_crops:
        mandi_crop = crop.capitalize()
        subset = df_forecasts[(df_forecasts["Mandi"].str.contains(mandi, case=False, na=False)) & 
                              (df_forecasts["Commodity"].str.contains(mandi_crop, case=False, na=False))]
                              
        if not subset.empty:
            max_price_14_days = subset['pred_modal_price'].max() if 'pred_modal_price' in subset.columns else subset.iloc[:,-1].max()
            print(f"   📈 {mandi_crop}: Expected Peak Price roughly ₹{max_price_14_days:.2f} / quintal")
            if max_price_14_days > highest_price:
                highest_price = max_price_14_days
                best_crop = mandi_crop
        else:
            print(f"   🤷 {mandi_crop}: No historical/predicted market data for this Mandi.")
            
    print("\n==========================================================")
    if highest_price > 0:
        print(f"🏆 FINAL HYBRID RECOMMENDATION: **{best_crop.upper()}**")
        print(f"   (It has strong agronomic feasibility AND highest expected selling price of ₹{highest_price:.2f}/quintal)")
    else:
        print(f"🏆 FINAL RECOMMENDATION: **{top_3_crops[0].upper()}** (Based entirely on biological conditions)")
    print("==========================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Crop Recommender (Biology + Economics)")
    parser.add_argument("--mandi", required=True, type=str, help="Name of target Mandi/Market (e.g. Pune, Kolar)")
    parser.add_argument("--n", type=float, default=90, help="Nitrogen ratio in soil")
    parser.add_argument("--p", type=float, default=42, help="Phosphorous ratio in soil")
    parser.add_argument("--k", type=float, default=43, help="Potassium ratio in soil")
    parser.add_argument("--ph", type=float, default=6.5, help="Soil pH level")
    parser.add_argument("--temp", type=float, default=25.0, help="Average Temperature (C)")
    parser.add_argument("--humidity", type=float, default=80.0, help="Relative Humidity (%)")
    parser.add_argument("--rainfall", type=float, default=200.0, help="Rainfall (mm)")
    
    args = parser.parse_args()
    recommend_best_profitable_crop(args.mandi, args.n, args.p, args.k, args.temp, args.humidity, args.ph, args.rainfall)
