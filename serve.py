#!/usr/bin/env python3
"""
AgriSense AI — Unified Dashboard Server
========================================
Single-file, zero-framework Python server.
Modules:
  1. Crop Recommendation  (LGBM model, 45 features)
  2. Market Forecaster    (Raw data + TFT forecasts)
  3. Plant Disease Scanner (Color-histogram heuristic)

Run:  python3 serve.py
Open: http://localhost:8080
"""

import json, io, os, sys, base64, traceback
import http.server, urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ── Paths ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
CROP_DATA = BASE / "CropRecommendationSystem" / "data" / "processed"
MKT_RAW   = Path("/Users/karthikreddy/Downloads/Crop_market_price_predictor/Agriculture_price_dataset.csv")
TFT_CSV   = BASE / "MandiPricePredictionSystem" / "data" / "processed" / "tft_forecasts.csv"
DL_CSV    = BASE / "MandiPricePredictionSystem" / "data" / "processed" / "dl_30_features_data.csv"

# ── Load models & data at startup ────────────────────────────────
print("Loading crop model...")
crop_model  = joblib.load(MODEL_DIR / "crop_recommendation_lgbm.pkl")
crop_scaler = joblib.load(MODEL_DIR / "crop_feature_scaler.pkl")
crop_enc    = joblib.load(CROP_DATA / "label_encoders.pkl")
crop_ref    = pd.read_parquet(CROP_DATA / "crop_features_engineered.parquet")
crop_classes = list(crop_enc['crop_type'].classes_)
crop_feats   = [c for c in crop_ref.columns if c not in ['crop_type','crop_type_encoded']]
print(f"  ✓ Crop model loaded ({len(crop_classes)} classes, {len(crop_feats)} features)")


# ── Load Disease Model ───────────────────────────────────────────────────────────
print("Loading Deep Learning Pathology model...")
try:
    disease_model = tf.keras.models.load_model(MODEL_DIR / 'best_model.keras')
    print("  ✓ Disease model loaded (EfficientNetB0)")
except Exception as e:
    disease_model = None
    print(f"  × Disease model not found: {e}")

try:
    with open(MODEL_DIR / 'classes.json', 'r') as f:
        disease_classes = json.load(f)
except:
    disease_classes = [f"Class_{i}" for i in range(38)]

print("Loading market data...")
raw_df = pd.read_csv(MKT_RAW)
raw_df['STATE'] = raw_df['STATE'].str.strip()
raw_df['Price Date'] = pd.to_datetime(raw_df['Price Date'], errors='coerce')
tft_df = pd.read_csv(TFT_CSV) if TFT_CSV.exists() else pd.DataFrame()
if not tft_df.empty:
    tft_df['date'] = pd.to_datetime(tft_df['date'])
# Build lookup indexes
states = sorted(raw_df['STATE'].dropna().unique().tolist())
state_mandis = {}
for s in states:
    state_mandis[s] = sorted(raw_df[raw_df['STATE']==s]['Market Name'].dropna().unique().tolist())
print(f"  ✓ Market data loaded ({len(raw_df)} rows, {len(states)} states)")

# ── Feature engineering (must match train_crop_model.py) ─────────
def engineer_crop_features(n, p, k, temp, hum, ph, rain):
    eps = 1e-6
    npk = n + p + k
    row = {
        'nitrogen': n, 'phosphorous': p, 'potassium': k,
        'temperature': temp, 'humidity': hum, 'ph': ph, 'rainfall': rain,
        'npk_total': npk, 'n_ratio': n/(npk+eps), 'p_ratio': p/(npk+eps), 'k_ratio': k/(npk+eps),
        'n_to_p': n/(p+eps), 'n_to_k': n/(k+eps), 'p_to_k': p/(k+eps),
        'n_times_p': n*p, 'n_times_k': n*k, 'p_times_k': p*k,
        'npk_product': n*p*k,
        'dominant_nutrient': 0 if (n>=p and n>=k) else (1 if (p>=n and p>=k) else 2),
        'nutrient_balance': float(np.std([n,p,k])),
        'nutrient_cv': float(np.std([n,p,k]) / (np.mean([n,p,k])+eps)),
        'heat_index': temp*hum/100.0,
        'rain_humidity_ratio': rain/(hum+eps),
        'aridity_index': temp/(rain+eps),
        'temp_x_rainfall': temp*rain, 'temp_x_humidity': temp*hum,
        'humidity_x_rainfall': hum*rain,
        'ph_x_nitrogen': ph*n, 'ph_x_rainfall': ph*rain,
        'ph_deviation_from_neutral': abs(ph-7.0), 'ph_squared': ph**2,
        'temp_squared': temp**2, 'humidity_squared': hum**2,
        'rainfall_squared': rain**2, 'rainfall_log': float(np.log1p(rain)),
        'rainfall_regime': 0 if rain<=50 else (1 if rain<=100 else (2 if rain<=150 else 3)),
        'ph_regime': 0 if ph<=5.5 else (1 if ph<=6.5 else (2 if ph<=7.5 else 3)),
        'temp_regime': 0 if temp<=20 else (1 if temp<=25 else (2 if temp<=30 else 3)),
    }
    for col, val in [('temperature',temp),('rainfall',rain),('nitrogen',n),
                     ('phosphorous',p),('potassium',k),('ph',ph),('humidity',hum)]:
        row[f'{col}_zscore'] = (val - crop_ref[col].mean()) / (crop_ref[col].std()+eps)
    df = pd.DataFrame([row])
    for f in crop_feats:
        if f not in df.columns:
            df[f] = 0
    return df[crop_feats]

# ── Disease detection via color analysis ─────────────────────────
def analyze_leaf(img_bytes):
    """Analyze leaf image using Deep Learning Keras Model."""
    if not disease_model:
        return {"error": "Deep Learning model not loaded. Please train the model first."}
    
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    arr = img_to_array(img) / 255.0  # normalize
    arr = np.expand_dims(arr, axis=0)
    
    probs = disease_model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100
    predicted_class = disease_classes[top_idx]
    
    # Generic mapping since it's deep learning categorical output
    is_healthy = "healthy" in predicted_class.lower()
    
    return {
        "disease": predicted_class.replace("_", " "),
        "confidence": round(confidence, 2),
        "severity": "None" if is_healthy else "Requires Attention",
        "assessment": f"Deep Learning pipeline classified image as {predicted_class} with {confidence:.1f}% confidence.",
        "prescriptions": [
            "Refer to agricultural extension for specific pathological treatment.",
            "Ensure proper watering and nutritional baseline."
        ]
    }

# ── Fertilizer recommendation ───────────────────────────────────
def get_fert_tips(n, p, k):
    tips = []
    if n < 20:   tips.append("🔴 Nitrogen LOW — Apply Urea (46-0-0) at 50-80 kg/ha")
    elif n < 40: tips.append("🟡 Nitrogen moderate — Light Urea top-dressing")
    else:        tips.append("🟢 Nitrogen adequate")
    if p < 20:   tips.append("🔴 Phosphorus LOW — Apply DAP (18-46-0)")
    elif p < 50: tips.append("🟡 Phosphorus moderate — Light DAP at sowing")
    else:        tips.append("🟢 Phosphorus adequate")
    if k < 20:   tips.append("🔴 Potassium LOW — Apply MOP (0-0-60)")
    elif k < 50: tips.append("🟡 Potassium moderate — Light MOP application")
    else:        tips.append("🟢 Potassium adequate")
    return tips

# ── HTML ─────────────────────────────────────────────────────────
HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriSense AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#f0f4f0;color:#1a1a2e;min-height:100vh}
.header{background:linear-gradient(135deg,#064e3b,#065f46,#047857);padding:24px 40px;color:#fff;
  display:flex;align-items:center;gap:16px;box-shadow:0 2px 12px rgba(0,0,0,.15)}
.header h1{font-size:28px;font-weight:800;letter-spacing:-.5px}
.header p{opacity:.8;font-size:14px;margin-top:2px}
.tabs{display:flex;gap:0;background:#fff;border-bottom:2px solid #e5e7eb;padding:0 40px;
  box-shadow:0 1px 3px rgba(0,0,0,.04)}
.tab{padding:14px 28px;cursor:pointer;font-weight:600;font-size:14px;color:#6b7280;
  border-bottom:3px solid transparent;transition:all .2s;user-select:none}
.tab:hover{color:#059669;background:#f0fdf4}
.tab.active{color:#059669;border-bottom-color:#10b981;background:#f0fdf4}
.content{max-width:1200px;margin:30px auto;padding:0 24px}
.panel{display:none}
.panel.active{display:block}
.card{background:#fff;border-radius:12px;padding:28px;margin-bottom:20px;
  box-shadow:0 1px 4px rgba(0,0,0,.06);border:1px solid #e5e7eb}
.card h3{font-size:18px;font-weight:700;margin-bottom:4px;color:#064e3b}
.card p.sub{color:#6b7280;font-size:13px;margin-bottom:20px}
.row{display:flex;gap:20px;flex-wrap:wrap}
.col{flex:1;min-width:240px}
label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:4px;margin-top:12px}
input[type=range]{width:100%;accent-color:#10b981}
.range-val{font-size:12px;color:#6b7280;text-align:right}
select{width:100%;padding:10px 12px;border:1px solid #d1d5db;border-radius:8px;font-size:14px;
  background:#fff;font-family:inherit;color:#1f2937}
select:focus{outline:none;border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.15)}
.btn{background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;padding:12px 32px;
  border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;transition:all .15s;
  display:inline-flex;align-items:center;gap:8px;margin-top:16px}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(16,185,129,.35)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* Results */
.result{margin-top:20px;padding:24px;border-radius:10px;border-left:5px solid #10b981;background:#f0fdf4}
.result.error{border-left-color:#ef4444;background:#fef2f2}
.result.warn{border-left-color:#f59e0b;background:#fffbeb}
.result h4{font-size:20px;margin-bottom:8px}
.result .conf{color:#6b7280;font-size:14px;font-weight:600}

.metric-row{display:flex;gap:16px;margin-top:16px;flex-wrap:wrap}
.metric{flex:1;min-width:160px;background:#fff;border-radius:10px;padding:20px;text-align:center;
  border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.metric .label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#6b7280;font-weight:600}
.metric .value{font-size:28px;font-weight:800;margin:6px 0;color:#064e3b}
.metric .delta{font-size:13px;font-weight:600}
.metric .delta.up{color:#10b981}
.metric .delta.down{color:#ef4444}

.alt-list{list-style:none;padding:0;margin-top:12px}
.alt-list li{padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:14px}
.alt-list li:last-child{border:none}
.tip{font-size:13px;padding:6px 0;color:#374151}

/* Disease */
.upload-zone{border:2px dashed #d1d5db;border-radius:12px;padding:40px;text-align:center;
  cursor:pointer;transition:all .2s;background:#fafafa}
.upload-zone:hover{border-color:#10b981;background:#f0fdf4}
.upload-zone.dragover{border-color:#10b981;background:#ecfdf5}
.upload-zone input{display:none}
.preview-img{max-width:100%;max-height:300px;border-radius:8px;margin-top:12px;object-fit:contain}
.severity-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;margin-left:8px}
.severity-badge.Severe{background:#fecaca;color:#991b1b}
.severity-badge.Moderate{background:#fef3c7;color:#92400e}
.severity-badge.Mild{background:#d1fae5;color:#065f46}
.severity-badge.None{background:#d1fae5;color:#065f46}
.rx-list{margin-top:12px;padding-left:20px}
.rx-list li{padding:4px 0;font-size:14px;color:#1f2937}
.spinner{display:inline-block;width:18px;height:18px;border:3px solid #fff;border-top-color:transparent;
  border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Chart */
.chart-container{margin-top:16px;background:#fff;border-radius:10px;padding:20px;
  border:1px solid #e5e7eb;overflow-x:auto}
table.forecast{width:100%;border-collapse:collapse;font-size:13px}
table.forecast th{background:#f9fafb;padding:10px;text-align:left;border-bottom:2px solid #e5e7eb;
  font-weight:600;color:#374151}
table.forecast td{padding:8px 10px;border-bottom:1px solid #f3f4f6}
table.forecast tr:hover{background:#f0fdf4}

@media(max-width:768px){
  .header{padding:16px 20px}.tabs{padding:0 12px;overflow-x:auto}
  .content{padding:0 12px;margin:16px auto}.row{flex-direction:column}
  .metric-row{flex-direction:column}.col{min-width:auto}
}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🌾 AgriSense AI</h1>
    <p>Crop Recommendation · Market Intelligence · Plant Pathology</p>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">🌱 Crop Recommendation</div>
  <div class="tab" onclick="switchTab(1)">📈 Market Forecaster</div>
  <div class="tab" onclick="switchTab(2)">🦠 Disease Scanner</div>
</div>

<div class="content">

<!-- ═══════ TAB 1: CROP ═══════ -->
<div class="panel active" id="panel-0">
  <div class="card">
    <h3>Biological Crop Recommender</h3>
    <p class="sub">Predicts the optimal crop using 45 climate, soil, and nutrient features via LightGBM.</p>
    <div class="row">
      <div class="col">
        <label>Nitrogen (N): <span id="v_n">50</span></label>
        <input type="range" id="n" min="0" max="140" value="50" oninput="v_n.textContent=this.value">
        <label>Phosphorous (P): <span id="v_p">50</span></label>
        <input type="range" id="p" min="0" max="145" value="50" oninput="v_p.textContent=this.value">
        <label>Potassium (K): <span id="v_k">50</span></label>
        <input type="range" id="k" min="0" max="205" value="50" oninput="v_k.textContent=this.value">
        <label>pH: <span id="v_ph">6.5</span></label>
        <input type="range" id="ph" min="3" max="10" value="6.5" step="0.1" oninput="v_ph.textContent=this.value">
      </div>
      <div class="col">
        <label>Temperature (°C): <span id="v_temp">25</span></label>
        <input type="range" id="temp" min="8" max="45" value="25" step="0.5" oninput="v_temp.textContent=this.value">
        <label>Humidity (%): <span id="v_hum">65</span></label>
        <input type="range" id="hum" min="10" max="100" value="65" oninput="v_hum.textContent=this.value">
        <label>Rainfall (mm): <span id="v_rain">100</span></label>
        <input type="range" id="rain" min="0" max="350" value="100" step="1" oninput="v_rain.textContent=this.value">
      </div>
    </div>
    <button class="btn" onclick="predictCrop()" id="cropBtn">🔬 Predict Best Crop</button>
    <div id="cropResult"></div>
  </div>
</div>

<!-- ═══════ TAB 2: MARKET ═══════ -->
<div class="panel" id="panel-1">
  <div class="card">
    <h3>📈 Hybrid Market Forecaster</h3>
    <p class="sub">Historical prices + 14-day TFT (Temporal Fusion Transformer) trajectory forecasts.</p>
    <div class="row">
      <div class="col">
        <label>State</label>
        <select id="mkt_state" onchange="loadMandis()"></select>
        <label>Market (Mandi)</label>
        <select id="mkt_mandi" onchange="loadCommodities()"></select>
      </div>
      <div class="col">
        <label>Commodity</label>
        <select id="mkt_commodity"></select>
      </div>
    </div>
    <button class="btn" onclick="queryMarket()" id="mktBtn">🚀 Generate Forecast</button>
    <div id="mktResult"></div>
  </div>
</div>

<!-- ═══════ TAB 3: DISEASE ═══════ -->
<div class="panel" id="panel-2">
  <div class="card">
    <h3>🦠 Plant Pathology Scanner</h3>
    <p class="sub">Upload a leaf image. Uses Deep Learning (EfficientNet-B0) to classify disease classifications.</p>
    <div class="row">
      <div class="col">
        <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
          <div style="font-size:36px;margin-bottom:8px">📷</div>
          <div style="font-weight:600;color:#374151">Click or drag an image here</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">JPG, PNG — max 10 MB</div>
          <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
        </div>
        <img id="previewImg" class="preview-img" style="display:none">
      </div>
      <div class="col">
        <div id="diseaseResult"></div>
      </div>
    </div>
  </div>
</div>

</div><!-- content -->

<script>
// ── Tabs ──
function switchTab(i){
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',j===i));
  document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('active',j===i));
}

// ── Crop ──
async function predictCrop(){
  const btn=document.getElementById('cropBtn');
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Processing...';
  try{
    const body={
      n:+document.getElementById('n').value,
      p:+document.getElementById('p').value,
      k:+document.getElementById('k').value,
      temp:+document.getElementById('temp').value,
      hum:+document.getElementById('hum').value,
      ph:+document.getElementById('ph').value,
      rain:+document.getElementById('rain').value
    };
    const r=await fetch('/api/crop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){document.getElementById('cropResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${d.error}</p></div>`;return}
    let html=`<div class="result"><h4>🏆 ${d.top_crop.charAt(0).toUpperCase()+d.top_crop.slice(1)}</h4>
      <div class="conf">Confidence: ${d.top_confidence}%</div>
      <div class="row" style="margin-top:16px"><div class="col">
        <strong>Top Alternatives:</strong><ul class="alt-list">`;
    d.alternatives.forEach(a=>{html+=`<li>${a.crop.charAt(0).toUpperCase()+a.crop.slice(1)} — ${a.confidence}%</li>`});
    html+=`</ul></div><div class="col"><strong>Fertilizer Action Plan:</strong>`;
    d.fertilizer_tips.forEach(t=>{html+=`<div class="tip">${t}</div>`});
    html+=`</div></div></div>`;
    document.getElementById('cropResult').innerHTML=html;
  }catch(e){document.getElementById('cropResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${e.message}</p></div>`}
  finally{btn.disabled=false;btn.innerHTML='🔬 Predict Best Crop'}
}

// ── Market ──
let marketMeta={};
async function initMarket(){
  const r=await fetch('/api/market/meta');
  marketMeta=await r.json();
  const sel=document.getElementById('mkt_state');
  sel.innerHTML=marketMeta.states.map(s=>`<option>${s}</option>`).join('');
  loadMandis();
}
async function loadMandis(){
  const st=document.getElementById('mkt_state').value;
  const r=await fetch('/api/market/mandis?state='+encodeURIComponent(st));
  const d=await r.json();
  document.getElementById('mkt_mandi').innerHTML=d.mandis.map(m=>`<option>${m}</option>`).join('');
  loadCommodities();
}
async function loadCommodities(){
  const st=document.getElementById('mkt_state').value;
  const mn=document.getElementById('mkt_mandi').value;
  const r=await fetch('/api/market/commodities?state='+encodeURIComponent(st)+'&mandi='+encodeURIComponent(mn));
  const d=await r.json();
  document.getElementById('mkt_commodity').innerHTML=d.commodities.map(c=>`<option>${c}</option>`).join('');
}
async function queryMarket(){
  const btn=document.getElementById('mktBtn');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Forecasting...';
  try{
    const body={
      state:document.getElementById('mkt_state').value,
      mandi:document.getElementById('mkt_mandi').value,
      commodity:document.getElementById('mkt_commodity').value
    };
    const r=await fetch('/api/market/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){document.getElementById('mktResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${d.error}</p></div>`;return}

    let html=`<div class="metric-row">
      <div class="metric"><div class="label">Latest Spot Price</div><div class="value">₹${d.current_price}</div><div style="font-size:12px;color:#6b7280">${d.price_date}</div></div>`;
    if(d.tft_target){
      const cls=d.tft_shift>=0?'up':'down';
      html+=`<div class="metric"><div class="label">14-Day TFT Target</div><div class="value">₹${d.tft_target}</div><div class="delta ${cls}">${d.tft_shift>=0?'+':''}${d.tft_shift}%</div></div>`;
    }
    html+=`<div class="metric"><div class="label">Records Found</div><div class="value">${d.record_count}</div></div></div>`;

    // Price history table
    if(d.history && d.history.length){
      html+=`<div class="chart-container"><strong>Recent Price History</strong>
        <table class="forecast"><tr><th>Date</th><th>Modal Price (₹)</th><th>Variety</th></tr>`;
      d.history.forEach(h=>{html+=`<tr><td>${h.date}</td><td>₹${h.price}</td><td>${h.variety}</td></tr>`});
      html+=`</table></div>`;
    }

    // TFT forecast table
    if(d.tft_forecast && d.tft_forecast.length){
      html+=`<div class="chart-container"><strong>TFT 14-Day Forecast Trajectory</strong>
        <table class="forecast"><tr><th>Date</th><th>Predicted Price (₹)</th></tr>`;
      d.tft_forecast.forEach(f=>{html+=`<tr><td>${f.date}</td><td style="font-weight:600;color:#059669">₹${f.price}</td></tr>`});
      html+=`</table></div>`;
    }

    document.getElementById('mktResult').innerHTML=html;
  }catch(e){document.getElementById('mktResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${e.message}</p></div>`}
  finally{btn.disabled=false;btn.innerHTML='🚀 Generate Forecast'}
}

// ── Disease ──
const dropZone=document.getElementById('dropZone');
['dragenter','dragover'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.add('dragover')}));
['dragleave','drop'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.remove('dragover')}));
dropZone.addEventListener('drop',ev=>{ if(ev.dataTransfer.files.length) handleFile(ev.dataTransfer.files[0]) });

async function handleFile(file){
  if(!file) return;
  // Preview
  const reader=new FileReader();
  reader.onload=e=>{
    const img=document.getElementById('previewImg');
    img.src=e.target.result; img.style.display='block';
  };
  reader.readAsDataURL(file);

  // Upload
  document.getElementById('diseaseResult').innerHTML='<div class="result" style="border-left-color:#6b7280"><h4><span class="spinner" style="border-color:#6b7280;border-top-color:transparent"></span> Analyzing leaf...</h4><p style="color:#6b7280;margin-top:8px">Running color-channel feature extraction...</p></div>';
  try{
    const fd=new FormData();
    fd.append('image',file);
    const r=await fetch('/api/disease',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){document.getElementById('diseaseResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${d.error}</p></div>`;return}

    const isHealthy=d.disease==='Healthy';
    const color=isHealthy?'#10b981':'#ef4444';
    const sevClass=d.severity.split(' ')[0];
    let html=`<div class="result" style="border-left-color:${color}">
      <h4 style="color:${color}">${isHealthy?'✅':'⚠️'} ${d.disease}${isHealthy?' — No Disease':' Detected'}
        <span class="severity-badge ${sevClass}">${d.severity}</span></h4>
      <div class="conf">Confidence: ${d.confidence}%</div>
      <p style="margin-top:12px"><strong>Assessment:</strong> ${d.assessment}</p>
      <p style="margin-top:12px"><strong>Prescriptions:</strong></p>
      <ul class="rx-list">`;
    d.prescriptions.forEach(p=>{html+=`<li>${p}</li>`});
    html+=`</ul></div>`;
    document.getElementById('diseaseResult').innerHTML=html;
  }catch(e){document.getElementById('diseaseResult').innerHTML=`<div class="result error"><h4>Error</h4><p>${e.message}</p></div>`}
}

// Init
initMarket();
</script>
</body>
</html>'''

# ── Server ───────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        body = HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = dict(urllib.parse.parse_qsl(parsed.query))

        if path == '/' or path == '/index.html':
            return self._html()

        elif path == '/api/market/meta':
            return self._json({'states': states})

        elif path == '/api/market/mandis':
            st = qs.get('state','')
            mandis = state_mandis.get(st, [])
            return self._json({'mandis': mandis})

        elif path == '/api/market/commodities':
            st = qs.get('state','')
            mn = qs.get('mandi','')
            sub = raw_df[(raw_df['STATE']==st)&(raw_df['Market Name']==mn)]
            coms = sorted(sub['Commodity'].dropna().unique().tolist())
            if not coms:
                coms = sorted(raw_df[raw_df['STATE']==st]['Commodity'].dropna().unique().tolist())
            return self._json({'commodities': coms})

        else:
            self.send_error(404)

    def do_POST(self):
        clen = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(clen)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == '/api/crop':
                data = json.loads(body)
                n,p,k = data['n'], data['p'], data['k']
                temp,hum,ph,rain = data['temp'],data['hum'],data['ph'],data['rain']

                df_in = engineer_crop_features(n,p,k,temp,hum,ph,rain)
                scaled = crop_scaler.transform(df_in)
                probs = crop_model.predict_proba(scaled)[0]
                top5 = np.argsort(probs)[::-1][:5]
                names = crop_enc['crop_type'].inverse_transform(top5)

                return self._json({
                    'top_crop': names[0],
                    'top_confidence': round(probs[top5[0]]*100, 1),
                    'alternatives': [{'crop':names[i],'confidence':round(probs[top5[i]]*100,1)} for i in range(1,5)],
                    'fertilizer_tips': get_fert_tips(n,p,k)
                })

            elif path == '/api/market/query':
                data = json.loads(body)
                st, mn, com = data['state'], data['mandi'], data['commodity']

                sub = raw_df[(raw_df['STATE']==st)&(raw_df['Market Name']==mn)&(raw_df['Commodity']==com)].copy()
                if sub.empty:
                    return self._json({'error': f'No records for {com} at {mn} ({st})'}, 404)

                sub = sub.sort_values('Price Date', ascending=False)
                latest = sub.iloc[0]
                cp = float(latest['Modal_Price'])
                pd_str = latest['Price Date'].strftime('%d %b %Y') if pd.notnull(latest['Price Date']) else 'Unknown'

                history = []
                for _, r in sub.head(10).iterrows():
                    history.append({
                        'date': r['Price Date'].strftime('%d %b %Y') if pd.notnull(r['Price Date']) else '—',
                        'price': int(r['Modal_Price']),
                        'variety': str(r.get('Variety','—'))
                    })

                # TFT
                tft_target = None
                tft_shift = None
                tft_list = []
                if not tft_df.empty:
                    tsub = tft_df[(tft_df['Mandi']==mn)&(tft_df['Commodity']==com)].sort_values('date')
                    if not tsub.empty:
                        tft_target = round(float(tsub.iloc[-1]['Predicted_ModalPrice']))
                        tft_shift = round((tft_target - cp) / (cp+1e-6) * 100, 1)
                        for _, r in tsub.iterrows():
                            tft_list.append({
                                'date': r['date'].strftime('%d %b %Y'),
                                'price': round(float(r['Predicted_ModalPrice']))
                            })

                return self._json({
                    'current_price': int(cp),
                    'price_date': pd_str,
                    'record_count': len(sub),
                    'history': history,
                    'tft_target': tft_target,
                    'tft_shift': tft_shift,
                    'tft_forecast': tft_list
                })

            elif path == '/api/disease':
                # Parse multipart form
                content_type = self.headers.get('Content-Type','')
                if 'multipart' not in content_type:
                    return self._json({'error':'Send multipart form data with image field'}, 400)

                boundary = content_type.split('boundary=')[1].encode()
                parts = body.split(b'--' + boundary)
                img_data = None
                for part in parts:
                    if b'name="image"' in part:
                        # Find start of binary data (after \r\n\r\n)
                        idx = part.find(b'\r\n\r\n')
                        if idx >= 0:
                            img_data = part[idx+4:]
                            # Remove trailing \r\n
                            if img_data.endswith(b'\r\n'):
                                img_data = img_data[:-2]
                        break

                if not img_data:
                    return self._json({'error':'No image found in upload'}, 400)

                result = analyze_leaf(img_data)
                return self._json(result)

            else:
                self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            return self._json({'error': str(e)}, 500)


if __name__ == '__main__':
    PORT = 8080
    server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  AgriSense AI running at http://localhost:{PORT}")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
