# CropRecommendationSystem/app.py
# ============================================================================
# AgriSense AI Dashboard (Streamlit)
# ============================================================================
# Production-grade interactive UI showcasing 45-feature ML pipeline:
#   - Real-time crop prediction with confidence scores
#   - Feature importance visualization
#   - Model comparison metrics display
#   - Top-3 crop alternatives with probability distribution
#   - Full input summary & engineered feature transparency
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AgriSense AI — Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "outputs" / "eval"


# ============================================================================
# Load all artifacts
# ============================================================================
@st.cache_resource
def load_pipeline():
    try:
        crop_model = joblib.load(MODEL_DIR / "crop_recommendation_lgbm.pkl")
        scaler = joblib.load(MODEL_DIR / "crop_feature_scaler.pkl")
        encoders = joblib.load(DATA_DIR / "label_encoders.pkl")
        df = pd.read_parquet(DATA_DIR / "crop_features_engineered.parquet")
        return crop_model, scaler, encoders, df
    except Exception as e:
        return None

@st.cache_data
def load_eval_data():
    try:
        summary = pd.read_csv(EVAL_DIR / "model_comparison_summary.csv")
        feat_imp = pd.read_csv(EVAL_DIR / "feature_importance_crop.csv")
        return summary, feat_imp
    except:
        return None, None


# ============================================================================
# Feature Engineering (must match build_features.py EXACTLY)
# ============================================================================
def engineer_single_input(n, p, k, temp, humidity, ph, rainfall, df_ref):
    """Build the full 45-feature vector for a single user input."""
    eps = 1e-6

    # --- Nutrient Interactions ---
    npk_total = n + p + k
    n_ratio = n / (npk_total + eps)
    p_ratio = p / (npk_total + eps)
    k_ratio = k / (npk_total + eps)
    n_to_p = n / (p + eps)
    n_to_k = n / (k + eps)
    p_to_k = p / (k + eps)
    n_times_p = n * p
    n_times_k = n * k
    p_times_k = p * k
    npk_product = n * p * k
    dominant_nutrient = 0 if (n >= p and n >= k) else (1 if (p >= n and p >= k) else 2)
    nutrient_balance = np.std([n, p, k])
    nutrient_cv = nutrient_balance / (np.mean([n, p, k]) + eps)

    # --- Climate Interactions ---
    heat_index = temp * humidity / 100.0
    rain_humidity_ratio = rainfall / (humidity + eps)
    aridity_index = temp / (rainfall + eps)
    temp_x_rainfall = temp * rainfall
    temp_x_humidity = temp * humidity
    humidity_x_rainfall = humidity * rainfall
    ph_x_nitrogen = ph * n
    ph_x_rainfall = ph * rainfall
    ph_deviation_from_neutral = abs(ph - 7.0)
    ph_squared = ph ** 2
    temp_squared = temp ** 2
    humidity_squared = humidity ** 2
    rainfall_squared = rainfall ** 2
    rainfall_log = np.log1p(rainfall)

    # --- Regime Bins ---
    # Rainfall regime
    if rainfall <= 50: rainfall_regime = 0
    elif rainfall <= 100: rainfall_regime = 1
    elif rainfall <= 150: rainfall_regime = 2
    elif rainfall <= 200: rainfall_regime = 3
    elif rainfall <= 300: rainfall_regime = 4
    else: rainfall_regime = 5

    # pH regime
    if ph <= 5.5: ph_regime = 0
    elif ph <= 6.5: ph_regime = 1
    elif ph <= 7.5: ph_regime = 2
    elif ph <= 8.5: ph_regime = 3
    else: ph_regime = 4

    # Temp regime
    if temp <= 20: temp_regime = 0
    elif temp <= 25: temp_regime = 1
    elif temp <= 30: temp_regime = 2
    elif temp <= 35: temp_regime = 3
    else: temp_regime = 4

    # --- Z-scores (global) ---
    zscore_features = {}
    for col_name, col_val in [
        ('temperature', temp), ('rainfall', rainfall), ('nitrogen', n),
        ('phosphorous', p), ('potassium', k), ('ph', ph), ('humidity', humidity)
    ]:
        col_mean = df_ref[col_name].mean()
        col_std = df_ref[col_name].std()
        zscore_features[f'{col_name}_zscore'] = (col_val - col_mean) / (col_std + eps)

    # --- Assemble Feature Vector ---
    row = {
        'nitrogen': n, 'phosphorous': p, 'potassium': k,
        'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
        'npk_total': npk_total,
        'n_ratio': n_ratio, 'p_ratio': p_ratio, 'k_ratio': k_ratio,
        'n_to_p': n_to_p, 'n_to_k': n_to_k, 'p_to_k': p_to_k,
        'n_times_p': n_times_p, 'n_times_k': n_times_k, 'p_times_k': p_times_k,
        'npk_product': npk_product, 'dominant_nutrient': dominant_nutrient,
        'nutrient_balance': nutrient_balance, 'nutrient_cv': nutrient_cv,
        'heat_index': heat_index, 'rain_humidity_ratio': rain_humidity_ratio,
        'aridity_index': aridity_index,
        'temp_x_rainfall': temp_x_rainfall, 'temp_x_humidity': temp_x_humidity,
        'humidity_x_rainfall': humidity_x_rainfall,
        'ph_x_nitrogen': ph_x_nitrogen, 'ph_x_rainfall': ph_x_rainfall,
        'ph_deviation_from_neutral': ph_deviation_from_neutral, 'ph_squared': ph_squared,
        'temp_squared': temp_squared, 'humidity_squared': humidity_squared,
        'rainfall_squared': rainfall_squared, 'rainfall_log': rainfall_log,
        'rainfall_regime': rainfall_regime, 'ph_regime': ph_regime, 'temp_regime': temp_regime,
        **zscore_features,
    }

    return pd.DataFrame([row])


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { background-color: #f8faf8; font-family: 'Inter', sans-serif; }
    .hero-box {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 30px 35px; border-radius: 16px; color: white;
        text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.15); margin-bottom: 20px;
    }
    .hero-box h1 { color: #b7e4c7; margin-bottom: 2px; font-size: 2.2em; }
    .hero-box h3 { color: #d8f3dc; font-weight: 400; margin-top: 0; }
    .metric-card {
        background: white; color: #333; padding: 18px 22px; border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); border-left: 5px solid #52b788;
        margin-bottom: 12px;
    }
    .metric-card h4 { margin: 0 0 4px 0; color: #1b4332; }
    .metric-card p { margin: 0; color: #555; font-size: 0.95em; }
    .alt-card {
        background: #f0f7f4; color: #333; padding: 14px 20px; border-radius: 10px;
        border: 1px solid #d8f3dc; margin-bottom: 8px;
    }
    .section-header {
        background: #f8f9fa; padding: 12px 20px; border-radius: 8px;
        border-left: 4px solid #2d6a4f; margin: 20px 0 10px 0;
        font-weight: 600; color: #1b4332;
    }
    .stat-box {
        background: white; padding: 15px; border-radius: 10px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-bottom: 3px solid #40916c;
    }
    .stat-box h2 { margin: 0; color: #2d6a4f; }
    .stat-box p { margin: 4px 0 0 0; color: #666; font-size: 0.85em; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2d6a4f, #40916c); color: white; border: none;
        border-radius: 10px; padding: 12px 28px; font-weight: 600; font-size: 1.05em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Dashboard Layout
# ============================================================================
pipeline = load_pipeline()

if pipeline is None:
    st.error("⚠️ Models not found. Run the pipeline first:")
    st.code("python build_features.py\npython train_crop_model.py", language="bash")
    st.stop()

crop_model, scaler, encoders, df_ref = pipeline
eval_summary, feat_imp = load_eval_data()
crop_le = encoders['crop_type']

# --- Hero ---
st.markdown("""
<div class="hero-box">
    <h1>🌾 AgriSense AI</h1>
    <h3>Intelligent Crop Recommendation Engine &nbsp;•&nbsp; 22 Crops &nbsp;•&nbsp; 45 Engineered Features</h3>
</div>
""", unsafe_allow_html=True)

# --- Pipeline Stats ---
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<div class="stat-box"><h2>22</h2><p>Crop Classes</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-box"><h2>45</h2><p>Engineered Features</p></div>', unsafe_allow_html=True)
with s3:
    best_acc = eval_summary['accuracy'].max() if eval_summary is not None else "N/A"
    st.markdown(f'<div class="stat-box"><h2>{best_acc}%</h2><p>Test Accuracy</p></div>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<div class="stat-box"><h2>{len(df_ref)}</h2><p>Training Samples</p></div>', unsafe_allow_html=True)

# --- Sidebar: Model Performance ---
with st.sidebar:
    st.markdown("## 📊 Model Benchmark")
    if eval_summary is not None:
        for _, row in eval_summary.iterrows():
            cv_info = ""
            if 'cv_accuracy_mean' in row and pd.notna(row.get('cv_accuracy_mean')):
                cv_info = f" | CV: {row['cv_accuracy_mean']}%±{row['cv_accuracy_std']}%"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['model']}</h4>
                <p>Acc: <b>{row['accuracy']}%</b> | F1: <b>{row['f1_macro']}</b>{cv_info}</p>
            </div>
            """, unsafe_allow_html=True)

    if feat_imp is not None:
        st.markdown("---")
        st.markdown("## 🏆 Top 10 Features")
        top_feats = feat_imp.head(10)
        st.bar_chart(top_feats.set_index('feature')['importance'])

    st.markdown("---")
    st.markdown("## 🔬 Pipeline Info")
    st.markdown("**Dataset:** Kaggle Crop Recommendation")
    st.markdown("**Models:** LightGBM vs RandomForest")
    st.markdown("**Split:** 80/20 Stratified")
    st.markdown("**CV:** 5-Fold Stratified")

# --- Main Inputs ---
st.markdown('<div class="section-header">🧪 Input Soil & Climate Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧬 Soil Nutrients (NPK)**")
    n = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content ratio in soil")
    p = st.slider("Phosphorous (P)", 0, 145, 50, help="Phosphorous content ratio in soil")
    k = st.slider("Potassium (K)", 0, 205, 50, help="Potassium content ratio in soil")

with col2:
    st.markdown("**🌤️ Climate Conditions**")
    temp = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, step=0.5)
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 65.0, step=1.0)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=5.0)

with col3:
    st.markdown("**🧪 Soil Chemistry**")
    ph = st.slider("Soil pH", 3.5, 10.0, 6.5, step=0.1, help="pH level of the soil (3.5=acidic, 10=alkaline)")
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Best Crop", use_container_width=True)


# --- Prediction ---
if predict_btn:
    with st.spinner("Running inference through 45 engineered features..."):
        # Build feature vector
        input_df = engineer_single_input(n, p, k, temp, humidity, ph, rainfall, df_ref)

        # Align columns to match training order
        drop_cols = ['crop_type', 'crop_type_encoded']
        train_features = [c for c in df_ref.columns if c not in drop_cols]

        # Ensure all columns exist (fill missing with 0)
        for col in train_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[train_features]

        # Scale
        input_scaled = scaler.transform(input_df.values)

        # Predict
        crop_probs = crop_model.predict_proba(input_scaled)[0]
        top_5_idx = np.argsort(crop_probs)[::-1][:5]
        top_5_crops = crop_le.inverse_transform(top_5_idx)
        top_5_conf = crop_probs[top_5_idx]

    # --- Results ---
    st.markdown('<div class="section-header">🏆 Recommendation Results</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1.5, 1])

    with r1:
        st.markdown(f"""
        <div class="hero-box" style="text-align: left;">
            <h3 style="color:#b7e4c7; margin-bottom:0;">Primary Recommendation</h3>
            <h1 style="font-size:2.5em; margin:8px 0;">🌱 {top_5_crops[0].title()}</h1>
            <p style="color:#d8f3dc; font-size:1.1em;">
                Confidence: <b>{top_5_conf[0]*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("**Top 5 Alternatives:**")
        for i in range(1, min(5, len(top_5_crops))):
            pct = top_5_conf[i] * 100
            st.markdown(f"""
            <div class="alt-card">
                <b>#{i+1}: {top_5_crops[i].title()}</b> — {pct:.1f}% feasibility
            </div>
            """, unsafe_allow_html=True)

    # Probability distribution chart
    st.markdown('<div class="section-header">📊 Full Probability Distribution Across All 22 Crops</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        'Crop': crop_le.classes_,
        'Probability (%)': crop_probs * 100
    }).sort_values('Probability (%)', ascending=True)
    st.bar_chart(prob_df.set_index('Crop'))

    # Input summary
    st.markdown('<div class="section-header">📋 Input & Engineered Features Summary</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw Inputs**")
        input_summary = pd.DataFrame([{
            'Nitrogen': n, 'Phosphorous': p, 'Potassium': k,
            'Temperature': f"{temp}°C", 'Humidity': f"{humidity}%",
            'pH': ph, 'Rainfall': f"{rainfall}mm",
        }])
        st.dataframe(input_summary, use_container_width=True)
    with c2:
        st.markdown("**Key Engineered Features**")
        eng_summary = pd.DataFrame([{
            'NPK Total': n + p + k,
            'N:P:K Ratio': f"{n}:{p}:{k}",
            'Heat Index': round(temp * humidity / 100, 2),
            'Aridity Index': round(temp / (rainfall + 1e-6), 4),
            'Nutrient Balance': round(np.std([n, p, k]), 2),
            'pH Deviation': round(abs(ph - 7.0), 2),
        }])
        st.dataframe(eng_summary, use_container_width=True)
