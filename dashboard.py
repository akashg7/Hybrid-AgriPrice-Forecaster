#!/usr/bin/env python3
"""
AgriSense AI — Unified Dashboard
Tabs: LightGBM Backtest | TFT Results | Crop Recommendation | Model Metrics
Run: python3 dashboard.py → http://localhost:8888
"""
import json, http.server, urllib.parse, warnings, os, re
import numpy as np, pandas as pd, joblib
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
HTML_PATH = BASE / "dashboard.html"
MKT_DIR = BASE / "MandiPricePredictionSystem"

# ══════════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ══════════════════════════════════════════════════════════════════
print("Loading LightGBM...")
lgbm = joblib.load(MKT_DIR / "models" / "price_lgbm_full_features.pkl")
print(f"  ✓ LightGBM ({lgbm.feature_name_})")

# ── DL features data (June 2025 – April 2026) ──
print("Loading price data...")
DL_CSV = MKT_DIR / "data" / "processed" / "dl_30_features_data.csv"
df = pd.read_csv(DL_CSV, usecols=["date","Mandi","Commodity","ModalPrice","target_price"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["Mandi","Commodity","date"])
print(f"  ✓ {len(df)} rows, {df['date'].min().date()} — {df['date'].max().date()}")
print(f"  ✓ {df['Mandi'].nunique()} mandis, {df['Commodity'].nunique()} commodities")

# ── State/District mapping from raw CSV ──
RAW_CSV = Path("/Users/karthikreddy/Downloads/Crop_market_price_predictor/Agriculture_price_dataset.csv")
state_map = {}  # dl_mandi_name → (state, district)
if RAW_CSV.exists():
    raw = pd.read_csv(RAW_CSV, usecols=["STATE","District Name","Market Name"])
    raw["STATE"] = raw["STATE"].str.strip()
    raw_lu = raw[["STATE","District Name","Market Name"]].drop_duplicates()
    raw_lu["key"] = raw_lu["Market Name"].str.strip().str.lower()
    raw_dict = {}
    for _,r in raw_lu.iterrows():
        raw_dict[r["key"]] = (r["STATE"], r["District Name"])
    def clean_mandi(s):
        s = re.sub(r'\s*APMC\s*$','',s,flags=re.IGNORECASE).strip()
        s = re.sub(r'\s*\(F&V\)\s*','',s,flags=re.IGNORECASE).strip()
        s = re.sub(r'\s*\(Rythu Bazar\)\s*','',s,flags=re.IGNORECASE).strip()
        return s.lower()
    for m in df["Mandi"].unique():
        k = clean_mandi(m)
        if k in raw_dict:
            state_map[m] = raw_dict[k]
    print(f"  ✓ State mapping: {len(state_map)}/{df['Mandi'].nunique()} mandis matched")

# Build hierarchy: state → district → mandis
hierarchy = {}  # state → {district → [mandis]}
for mandi, (state, dist) in state_map.items():
    if state not in hierarchy: hierarchy[state] = {}
    if dist not in hierarchy[state]: hierarchy[state][dist] = []
    hierarchy[state][dist].append(mandi)
for s in hierarchy:
    for d in hierarchy[s]:
        hierarchy[s][d] = sorted(set(hierarchy[s][d]))
states = sorted(hierarchy.keys())

# Mandi → commodities
mandi_coms = {}
for m in df["Mandi"].unique():
    coms = sorted(df[df["Mandi"]==m]["Commodity"].unique().tolist())
    if coms: mandi_coms[m] = coms

# ══════════════════════════════════════════════════════════════════
# 2. LGBM BACKTEST: predict from 14 days before end, compare actual
# ══════════════════════════════════════════════════════════════════
def lgbm_backtest(mandi, commodity):
    sub = df[(df["Mandi"]==mandi)&(df["Commodity"]==commodity)].copy()
    sub = sub.sort_values("date").reset_index(drop=True)
    if len(sub) < 21:
        return {"error": f"Only {len(sub)} rows for {commodity} at {mandi} (need 21+)"}
    
    # Use data up to 14 days before end as "known", predict last 14 days
    cutoff_idx = len(sub) - 14
    train_part = sub.iloc[:cutoff_idx]
    test_part = sub.iloc[cutoff_idx:]
    
    prices = train_part["ModalPrice"].values[-7:]
    last_date = train_part["date"].iloc[-1]
    
    # History (last 30 known days)
    hist = train_part.tail(30)
    history = [[r["date"].strftime("%Y-%m-%d"), float(r["ModalPrice"])] for _,r in hist.iterrows()]
    
    # Iterative 14-day forecast
    forecast = []
    actuals = []
    buffer = list(prices)
    for i in range(min(14, len(test_part))):
        actual_row = test_part.iloc[i]
        fdate = actual_row["date"]
        lag1, lag2, lag3 = buffer[-1], buffer[-2], buffer[-3]
        lag7 = buffer[-7] if len(buffer) >= 7 else buffer[0]
        X = np.array([[lag1, lag2, lag3, lag7, fdate.dayofweek, fdate.month]])
        pred = max(float(lgbm.predict(X)[0]), 1.0)
        forecast.append([fdate.strftime("%Y-%m-%d"), round(pred, 2)])
        actuals.append([fdate.strftime("%Y-%m-%d"), float(actual_row["ModalPrice"])])
        buffer.append(pred)
    
    # Compute error metrics
    if actuals:
        y_true = np.array([a[1] for a in actuals])
        y_pred = np.array([f[1] for f in forecast])
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    else:
        mae, mape = None, None
    
    info = state_map.get(mandi, ("Unknown", "Unknown"))
    return {"history": history, "forecast": forecast, "actuals": actuals,
            "mandi": mandi, "commodity": commodity, "state": info[0], "district": info[1],
            "mae": round(mae, 2) if mae else None, "mape": round(mape, 2) if mape else None,
            "cutoff_date": last_date.strftime("%Y-%m-%d"),
            "data_end": sub["date"].iloc[-1].strftime("%Y-%m-%d")}

# ══════════════════════════════════════════════════════════════════
# 3. TFT PRE-COMPUTED RESULTS (from evaluate_tft.py output)
# ══════════════════════════════════════════════════════════════════
tft_eval = {
    "tft": {"mae":319.88,"rmse":647.88,"mape":15.77,"smape":13.41,"accuracy":86.59,"epoch":9,"steps":43310,"samples":3815},
    "naive": {"mae":344.36,"rmse":799.32,"mape":15.53,"smape":13.14,"accuracy":86.86},
}
# Load per-sequence results if available
tft_csv = MKT_DIR / "outputs" / "eval" / "eval_tft_checkpoint_actual.csv"
if tft_csv.exists():
    tft_detail = pd.read_csv(tft_csv)
    print(f"  ✓ TFT eval results loaded")

# ══════════════════════════════════════════════════════════════════
# 4. CROP RECOMMENDATION
# ══════════════════════════════════════════════════════════════════
print("Loading crop model...")
MDIR = BASE / "models"
crop_model = joblib.load(MDIR / "crop_recommendation_lgbm.pkl")
crop_scaler = joblib.load(MDIR / "crop_feature_scaler.pkl")
CDATA = BASE / "CropRecommendationSystem" / "data" / "processed"
crop_enc = joblib.load(CDATA / "label_encoders.pkl")
crop_ref = pd.read_parquet(CDATA / "crop_features_engineered.parquet")
crop_classes = list(crop_enc['crop_type'].classes_)
crop_feats = [c for c in crop_ref.columns if c not in ['crop_type','crop_type_encoded']]
print(f"  ✓ Crop model ({len(crop_classes)} classes)")

def engineer_crop(n,p,k,temp,hum,ph,rain):
    eps=1e-6; npk=n+p+k
    row = {'nitrogen':n,'phosphorous':p,'potassium':k,'temperature':temp,'humidity':hum,'ph':ph,'rainfall':rain,
        'npk_total':npk,'n_ratio':n/(npk+eps),'p_ratio':p/(npk+eps),'k_ratio':k/(npk+eps),
        'n_to_p':n/(p+eps),'n_to_k':n/(k+eps),'p_to_k':p/(k+eps),
        'n_times_p':n*p,'n_times_k':n*k,'p_times_k':p*k,'npk_product':n*p*k,
        'dominant_nutrient':0 if(n>=p and n>=k)else(1 if(p>=n and p>=k)else 2),
        'nutrient_balance':float(np.std([n,p,k])),'nutrient_cv':float(np.std([n,p,k])/(np.mean([n,p,k])+eps)),
        'heat_index':temp*hum/100.0,'rain_humidity_ratio':rain/(hum+eps),'aridity_index':temp/(rain+eps),
        'temp_x_rainfall':temp*rain,'temp_x_humidity':temp*hum,'humidity_x_rainfall':hum*rain,
        'ph_x_nitrogen':ph*n,'ph_x_rainfall':ph*rain,'ph_deviation_from_neutral':abs(ph-7.0),'ph_squared':ph**2,
        'temp_squared':temp**2,'humidity_squared':hum**2,'rainfall_squared':rain**2,'rainfall_log':float(np.log1p(rain)),
        'rainfall_regime':0 if rain<=50 else(1 if rain<=100 else(2 if rain<=150 else 3)),
        'ph_regime':0 if ph<=5.5 else(1 if ph<=6.5 else(2 if ph<=7.5 else 3)),
        'temp_regime':0 if temp<=20 else(1 if temp<=25 else(2 if temp<=30 else 3))}
    for col,val in [('temperature',temp),('rainfall',rain),('nitrogen',n),('phosphorous',p),('potassium',k),('ph',ph),('humidity',hum)]:
        row[f'{col}_zscore']=(val-crop_ref[col].mean())/(crop_ref[col].std()+eps)
    d=pd.DataFrame([row])
    for f in crop_feats:
        if f not in d.columns: d[f]=0
    return d[crop_feats]

def fert_tips(n,p,k):
    t=[]
    t.append("🔴 N LOW — Apply Urea 50-80 kg/ha" if n<20 else("🟡 N moderate" if n<40 else "🟢 N adequate"))
    t.append("🔴 P LOW — Apply DAP" if p<20 else("🟡 P moderate" if p<50 else "🟢 P adequate"))
    t.append("🔴 K LOW — Apply MOP" if k<20 else("🟡 K moderate" if k<50 else "🟢 K adequate"))
    return t

# ══════════════════════════════════════════════════════════════════
# 5. SERVER
# ══════════════════════════════════════════════════════════════════
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _j(self,d,c=200):
        b=json.dumps(d,default=str).encode()
        self.send_response(c);self.send_header('Content-Type','application/json');self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def _h(self,html):
        b=html.encode();self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=urllib.parse.urlparse(self.path);path=p.path;qs=dict(urllib.parse.parse_qsl(p.query))
        if path in ('/','/index.html'):
            self._h(HTML_PATH.read_text())
        elif path=='/api/states':
            self._j(states)
        elif path=='/api/districts':
            s=qs.get('state','')
            self._j(sorted(hierarchy.get(s,{}).keys()))
        elif path=='/api/mandis':
            s,d=qs.get('state',''),qs.get('district','')
            self._j(hierarchy.get(s,{}).get(d,[]))
        elif path=='/api/commodities':
            self._j(mandi_coms.get(qs.get('mandi',''),[]))
        elif path=='/api/tft_eval':
            self._j(tft_eval)
        else: self.send_error(404)
    def do_POST(self):
        body=self.rfile.read(int(self.headers.get('Content-Length',0)))
        path=urllib.parse.urlparse(self.path).path
        if path=='/api/lgbm/predict':
            d=json.loads(body);self._j(lgbm_backtest(d['mandi'],d['commodity']))
        elif path=='/api/crop':
            d=json.loads(body)
            X=engineer_crop(d['nitrogen'],d['phosphorous'],d['potassium'],d['temperature'],d['humidity'],d['ph'],d['rainfall'])
            X_s=crop_scaler.transform(X);probs=crop_model.predict_proba(X_s)[0]
            top5=np.argsort(probs)[::-1][:5];names=[crop_classes[i] for i in top5]
            self._j({'top_crop':names[0],'top_confidence':round(probs[top5[0]]*100,1),
                'alternatives':[{'crop':names[i],'confidence':round(probs[top5[i]]*100,1)} for i in range(1,5)],
                'fertilizer_tips':fert_tips(d['nitrogen'],d['phosphorous'],d['potassium'])})
        else: self.send_error(404)

if __name__=='__main__':
    PORT=8888; os.system(f"lsof -ti:{PORT} | xargs kill -9 2>/dev/null")
    srv=http.server.HTTPServer(('0.0.0.0',PORT),H)
    print(f"\n{'='*50}\n  AgriSense: http://localhost:{PORT}\n{'='*50}\n")
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
