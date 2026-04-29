#!/usr/bin/env python3
"""TFT Results Dashboard + Live Prediction Server. Run: python3 tft_app.py"""
import json, http.server, urllib.parse, warnings, os
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")

from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

DATA_PATH = Path("data/processed/dl_30_features_data.csv")
CKPT_PATH = Path("epoch=9-step=43310.ckpt")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["Mandi","Commodity","date"])
df["time_idx"] = df.groupby(["Mandi","Commodity"]).cumcount()
df["target_price"] = df["target_price"].clip(lower=1.0)
df["group_id"] = df["Mandi"].astype(str)+"_"+df["Commodity"].astype(str)
for col in ["temp_avg","humidity","rainfall","rolling_mean_7","volatility_7","momentum_7"]:
    if col in df.columns: df[col]=df[col].fillna(df[col].median())
df = df.dropna(subset=["target_price","temp_avg","humidity","rainfall","rolling_mean_7","volatility_7","momentum_7","day_of_year","sin1","cos1"])

max_pred=14; max_enc=30
cutoff = df["time_idx"].max()-max_pred

print("Building dataset...")
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx<=cutoff], time_idx="time_idx", target="target_price",
    group_ids=["group_id"], min_encoder_length=max_enc, max_encoder_length=max_enc,
    min_prediction_length=max_pred, max_prediction_length=max_pred,
    static_categoricals=["Mandi","Commodity"],
    time_varying_known_reals=["time_idx","day_of_year","sin1","cos1"],
    time_varying_unknown_reals=["target_price","temp_avg","humidity","rainfall","rolling_mean_7","volatility_7","momentum_7"],
    target_normalizer=GroupNormalizer(groups=["group_id"],transformation="softplus"),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
)

print("Loading TFT checkpoint...")
import torchmetrics
_orig = torchmetrics.Metric._apply
def _safe(self,fn,*a,**k): self._device=torch.device("cpu"); return torch.nn.Module._apply(self,fn)
torchmetrics.Metric._apply = _safe
raw = torch.load(str(CKPT_PATH),map_location="cpu",weights_only=False)
if "callbacks" in raw: raw["callbacks"]={}
tmp=str(CKPT_PATH)+".tmp"; torch.save(raw,tmp)
model = TemporalFusionTransformer.load_from_checkpoint(tmp,map_location="cpu")
os.remove(tmp)
torchmetrics.Metric._apply = _orig
model.eval()
print("✓ Model loaded")

# Build available pairs
pairs = df.groupby(["Mandi","Commodity"]).size().reset_index(name="count")
pairs = pairs[pairs["count"]>=max_enc+max_pred]
mandis = sorted(pairs["Mandi"].unique().tolist())
mandi_commodities = {}
for m in mandis:
    mandi_commodities[m] = sorted(pairs[pairs["Mandi"]==m]["Commodity"].unique().tolist())

def predict_pair(mandi, commodity):
    gid = f"{mandi}_{commodity}"
    sub = df[df["group_id"]==gid].copy()
    if len(sub) < max_enc+max_pred: return None
    sub = sub.sort_values("date")
    last_date = sub["date"].iloc[-1]
    # Get last encoder window
    encoder_data = sub.tail(max_enc+max_pred)
    try:
        pred_ds = TimeSeriesDataSet.from_dataset(training, encoder_data, predict=True, stop_randomization=True)
        dl = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw_out = model.predict(dl, mode="raw")
        # Get median (quantile index 3 of 7)
        if hasattr(raw_out, 'output'):
            preds = raw_out.output[0][0,:,3].detach().numpy()
        else:
            preds = model.predict(dl, mode="prediction")[0].detach().numpy()
        # Get recent history
        hist = sub.tail(30)[["date","ModalPrice"]].copy()
        hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
        history = hist.values.tolist()
        # Build forecast dates
        forecast = []
        for i,p in enumerate(preds):
            d = last_date + pd.Timedelta(days=i+1)
            forecast.append([d.strftime("%Y-%m-%d"), round(float(p),2)])
        return {"history": history, "forecast": forecast, "mandi": mandi, "commodity": commodity}
    except Exception as e:
        return {"error": str(e)}

# Eval results
EVAL = {
    "tft": {"mae":319.88,"rmse":647.88,"mape":15.77,"smape":13.41,"accuracy":86.59,"samples":3815,"epoch":9,"steps":43310},
    "naive": {"mae":344.36,"rmse":799.32,"mape":15.53,"smape":13.14,"accuracy":86.86},
    "lgbm": {"mae":160.41,"rmse":355.80,"mape":10.77,"smape":8.80,"accuracy":91.20,"samples":56826},
    "naive_global": {"mae":1284.38,"rmse":1757.20,"mape":55.19,"smape":78.66,"accuracy":21.34}
}

HTML = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TFT Results & Prediction — AgriSense AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#1e293b 0%,#0f172a 50%,#1e1b4b 100%);padding:32px 40px;border-bottom:1px solid #334155}
.header h1{font-size:32px;font-weight:800;background:linear-gradient(135deg,#10b981,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header p{color:#94a3b8;margin-top:4px;font-size:14px}
.tabs{display:flex;gap:0;background:#1e293b;border-bottom:1px solid #334155;padding:0 40px}
.tab{padding:14px 28px;cursor:pointer;font-weight:600;font-size:14px;color:#64748b;border-bottom:3px solid transparent;transition:.2s}
.tab:hover{color:#10b981}.tab.active{color:#10b981;border-bottom-color:#10b981;background:#0f172a22}
.content{max-width:1200px;margin:30px auto;padding:0 24px}
.panel{display:none}.panel.active{display:block}
.card{background:#1e293b;border-radius:16px;padding:28px;margin-bottom:20px;border:1px solid #334155;box-shadow:0 4px 24px rgba(0,0,0,.3)}
.card h3{font-size:20px;font-weight:700;margin-bottom:4px;color:#f1f5f9}
.card .sub{color:#94a3b8;font-size:13px;margin-bottom:20px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0}
.metric{background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:12px;padding:20px;text-align:center;border:1px solid #334155}
.metric .label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;font-weight:600}
.metric .value{font-size:28px;font-weight:800;margin:8px 0}
.metric .value.green{color:#10b981}.metric .value.blue{color:#3b82f6}.metric .value.amber{color:#f59e0b}.metric .value.red{color:#ef4444}
.metric .note{font-size:11px;color:#64748b}
table{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}
th{background:#0f172a;padding:12px;text-align:left;border-bottom:2px solid #334155;color:#94a3b8;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:1px}
td{padding:10px 12px;border-bottom:1px solid #1e293b66}
tr:hover{background:#ffffff08}
.best{color:#10b981;font-weight:700}
.row{display:flex;gap:20px;flex-wrap:wrap}.col{flex:1;min-width:240px}
select{width:100%;padding:10px 12px;border:1px solid #334155;border-radius:8px;font-size:14px;background:#0f172a;color:#e2e8f0;font-family:inherit}
select:focus{outline:none;border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.2)}
label{display:block;font-size:13px;font-weight:600;color:#94a3b8;margin:12px 0 4px}
.btn{background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;padding:12px 32px;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;transition:.15s;margin-top:16px}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(16,185,129,.4)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.chart{position:relative;width:100%;height:300px;margin:20px 0;background:#0f172a;border-radius:12px;border:1px solid #334155;overflow:hidden}
canvas{width:100%!important;height:100%!important}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;margin-right:6px}
.tag.green{background:#064e3b;color:#10b981}.tag.blue{background:#1e3a5f;color:#3b82f6}.tag.amber{background:#451a03;color:#f59e0b}
.lit{background:#0f172a;border-radius:8px;padding:16px;margin:12px 0;border-left:3px solid #3b82f6}
.lit h4{color:#3b82f6;font-size:14px;margin-bottom:4px}
.lit p{color:#94a3b8;font-size:13px;line-height:1.6}
.spinner{display:inline-block;width:18px;height:18px;border:3px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.forecast-table td.pred{color:#10b981;font-weight:700;font-size:15px}
.alert{padding:16px;border-radius:8px;margin:16px 0;font-size:13px}
.alert.info{background:#1e3a5f;border:1px solid #2563eb;color:#93c5fd}
.alert.warn{background:#451a03;border:1px solid #d97706;color:#fcd34d}
@media(max-width:768px){.header{padding:16px 20px}.tabs{padding:0 12px;overflow-x:auto}.content{padding:0 12px}.row{flex-direction:column}.metrics{grid-template-columns:1fr 1fr}}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head><body>
<div class="header">
<h1>🌾 TFT Results & Prediction</h1>
<p>Temporal Fusion Transformer — Verified Checkpoint Evaluation & Live 14-Day Forecasting</p>
</div>
<div class="tabs">
<div class="tab active" onclick="switchTab(0)">📊 Evaluation Results</div>
<div class="tab" onclick="switchTab(1)">🔮 Live Prediction</div>
<div class="tab" onclick="switchTab(2)">📚 Methodology & Literature</div>
</div>
<div class="content">

<!-- TAB 1: RESULTS -->
<div class="panel active" id="p0">
<div class="card">
<h3>Model Comparison — Verified from Checkpoint</h3>
<p class="sub">All numbers computed via <code>evaluate_tft.py</code> using <code>epoch=9-step=43310.ckpt</code></p>
<div class="metrics" id="metricsGrid"></div>
</div>
<div class="card">
<h3>Regime A — Single-Step Prediction (LightGBM)</h3>
<p class="sub">56,826 test rows · Naive = one global constant for all mandis/crops</p>
<table><tr><th>Model</th><th>MAE (₹/qtl)</th><th>RMSE</th><th>SMAPE</th><th>Accuracy</th></tr>
<tr><td>Naive (global constant)</td><td>1,284.38</td><td>1,757.20</td><td>78.66%</td><td>21.34%</td></tr>
<tr><td class="best">LightGBM (full features)</td><td class="best">160.41</td><td class="best">355.80</td><td class="best">8.80%</td><td class="best">91.20%</td></tr>
</table>
</div>
<div class="card">
<h3>Regime B — 14-Day Multi-Horizon (TFT)</h3>
<p class="sub">3,815 validation sequences · Naive = last known price per mandi-commodity pair</p>
<table><tr><th>Model</th><th>MAE (₹/qtl)</th><th>RMSE</th><th>SMAPE</th><th>Accuracy</th></tr>
<tr><td>Naive (per-group carry-forward)</td><td>344.36</td><td>799.32</td><td>13.14%</td><td>86.86%</td></tr>
<tr><td>TFT (epoch 9, 43,310 steps)</td><td>319.88</td><td>647.88</td><td>13.41%</td><td>86.59%</td></tr>
</table>
<div class="alert warn">⚠️ TFT does not beat the naive carry-forward baseline on SMAPE. The model shows signs of both underfitting (can't beat naive) and overfitting (epoch 9 worse than epoch 6). Architecture is too small: hidden_size=16 with 54K params across 4,612 groups.</div>
</div>
<div class="card">
<h3>Checkpoint Details</h3>
<table><tr><th>Property</th><th>Value</th></tr>
<tr><td>File</td><td><code>epoch=9-step=43310.ckpt</code> (4.08 MB)</td></tr>
<tr><td>Parameters</td><td>54,029 (all non-zero)</td></tr>
<tr><td>Architecture</td><td>hidden_size=16, 1 attention head, dropout=0.1, 7 quantile outputs</td></tr>
<tr><td>Encoder / Decoder</td><td>30-day lookback → 14-day forecast</td></tr>
<tr><td>Loss</td><td>QuantileLoss</td></tr>
<tr><td>Framework</td><td>PyTorch Forecasting + Lightning</td></tr>
</table>
</div>
</div>

<!-- TAB 2: PREDICTION -->
<div class="panel" id="p1">
<div class="card">
<h3>🔮 Live TFT 14-Day Forecast</h3>
<p class="sub">Select a mandi-commodity pair. Runs actual inference from the trained checkpoint — each day gets a distinct predicted price.</p>
<div class="row">
<div class="col">
<label>Mandi (Market)</label>
<select id="sel_mandi" onchange="loadComs()"></select>
</div>
<div class="col">
<label>Commodity</label>
<select id="sel_com"></select>
</div>
</div>
<button class="btn" onclick="runPredict()" id="predBtn">🚀 Generate 14-Day Forecast</button>
<div id="predResult"></div>
</div>
</div>

<!-- TAB 3: LITERATURE -->
<div class="panel" id="p2">
<div class="card">
<h3>Architecture — Temporal Fusion Transformer</h3>
<p class="sub">Why TFT was chosen and what each component does</p>
<div class="alert info">TFT (Lim et al., 2021) is purpose-built for multi-horizon forecasting with heterogeneous inputs — static metadata, known future values, and observed past values. It combines Variable Selection Networks, Gated Residual Networks, and interpretable multi-head attention.</div>
<table><tr><th>Component</th><th>Purpose</th><th>Our Config</th></tr>
<tr><td>Variable Selection Networks</td><td>Auto-weight importance of each input feature</td><td>Mandi, Commodity (static) + weather, lags (observed)</td></tr>
<tr><td>Gated Residual Networks</td><td>Skip connections — act simple when data is simple</td><td>hidden_size=16 (too small for 4,612 groups)</td></tr>
<tr><td>Multi-Head Attention</td><td>Capture long-range temporal dependencies</td><td>1 head (recommended: 4+)</td></tr>
<tr><td>Quantile Output</td><td>Probabilistic forecasts (confidence bands)</td><td>7 quantiles</td></tr>
</table>
</div>
<div class="card">
<h3>📚 Literature Review</h3>
<div class="lit"><h4>Lim et al. (2021) — Temporal Fusion Transformers</h4><p>Int. J. Forecasting. Introduced TFT with VSN, GRN, and interpretable attention for heterogeneous time series. Recommended hidden_size=160-240 for production — our 16 is 10× smaller, explaining the underfitting.</p></div>
<div class="lit"><h4>Grinsztajn et al. (2022) — Tree-based vs Deep Learning</h4><p>NeurIPS. On medium tabular datasets with engineered features, gradient boosting consistently matches or beats deep learning. Our LightGBM (91.2%) vs TFT (86.6%) is consistent with this finding.</p></div>
<div class="lit"><h4>Woo et al. (2024) — Moirai Foundation Model</h4><p>ICML. Zero-shot time series transformers struggle on localized, covariate-heavy domains. Our Moirai result (MAE ~543) confirms domain-specific training is essential.</p></div>
<div class="lit"><h4>Makridakis et al. (2018) — Simple vs Complex Methods</h4><p>PLOS ONE. Simple methods (naive, exponential smoothing) often outperform complex ML on real-world time series. Our naive ≈ TFT result aligns with this well-documented phenomenon.</p></div>
<div class="lit"><h4>Box et al. (2015) — Time Series Analysis</h4><p>5th ed., Wiley. Agricultural prices exhibit high autocorrelation — the carry-forward naive baseline (87% accuracy) is inherently strong because today's price ≈ yesterday's price.</p></div>
<div class="lit"><h4>Chand (2012) & NITI Aayog (2019)</h4><p>Indian mandi price volatility: 200-400% swings in perishables (onion, tomato). Information asymmetry costs ~$8B annually in post-harvest losses. Motivates the need for forecasting systems.</p></div>
</div>
<div class="card">
<h3>What Would Improve TFT</h3>
<table><tr><th>Parameter</th><th>Current</th><th>Recommended</th></tr>
<tr><td>hidden_size</td><td>16</td><td>128-256</td></tr>
<tr><td>attention_heads</td><td>1</td><td>4-8</td></tr>
<tr><td>training epochs</td><td>9</td><td>30-50</td></tr>
<tr><td>encoder window</td><td>30 days</td><td>60-90 days</td></tr>
<tr><td>observed covariates</td><td>7</td><td>15+ (add all weather, arrivals)</td></tr>
</table>
</div>
</div>
</div>

<script>
function switchTab(i){document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',j===i));document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('active',j===i))}
const E=EVAL_DATA;
document.getElementById('metricsGrid').innerHTML=`
<div class="metric"><div class="label">LightGBM Accuracy</div><div class="value green">${E.lgbm.accuracy}%</div><div class="note">Best model · 56,826 rows</div></div>
<div class="metric"><div class="label">TFT Accuracy</div><div class="value blue">${E.tft.accuracy}%</div><div class="note">14-day horizon · 3,815 seq</div></div>
<div class="metric"><div class="label">TFT MAE</div><div class="value amber">₹${E.tft.mae}</div><div class="note">vs naive ₹${E.naive.mae}</div></div>
<div class="metric"><div class="label">TFT Epoch</div><div class="value blue">${E.tft.epoch}</div><div class="note">${E.tft.steps.toLocaleString()} steps</div></div>`;

let MC=MANDI_COMS;
let ms=Object.keys(MC).sort();
document.getElementById('sel_mandi').innerHTML=ms.map(m=>`<option>${m}</option>`).join('');
function loadComs(){let m=document.getElementById('sel_mandi').value;document.getElementById('sel_com').innerHTML=(MC[m]||[]).map(c=>`<option>${c}</option>`).join('')}
loadComs();

let chart=null;
async function runPredict(){
  const btn=document.getElementById('predBtn');btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Running TFT inference...';
  const m=document.getElementById('sel_mandi').value,c=document.getElementById('sel_com').value;
  try{
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mandi:m,commodity:c})});
    const d=await r.json();
    if(d.error){document.getElementById('predResult').innerHTML=`<div class="alert warn">⚠️ ${d.error}</div>`;return}
    // Build chart
    let labels=[...d.history.map(h=>h[0]),...d.forecast.map(f=>f[0])];
    let histPrices=d.history.map(h=>h[1]);
    let pad=new Array(d.history.length).fill(null);
    let fcPrices=[...pad,...d.forecast.map(f=>f[1])];
    // Connect: last historical point to first forecast
    fcPrices[d.history.length-1]=histPrices[histPrices.length-1];
    let histFull=[...histPrices,...new Array(d.forecast.length).fill(null)];
    if(chart)chart.destroy();
    document.getElementById('predResult').innerHTML=`<div class="chart"><canvas id="fcChart"></canvas></div>
      <table class="forecast-table"><tr><th>Date</th><th>Predicted Price (₹/qtl)</th></tr>${d.forecast.map(f=>`<tr><td>${f[0]}</td><td class="pred">₹${f[1].toLocaleString()}</td></tr>`).join('')}</table>`;
    const ctx=document.getElementById('fcChart').getContext('2d');
    chart=new Chart(ctx,{type:'line',data:{labels:labels.map(l=>l.slice(5)),datasets:[
      {label:'Historical Price',data:histFull,borderColor:'#64748b',backgroundColor:'#64748b22',fill:true,tension:.3,pointRadius:1},
      {label:'TFT 14-Day Forecast',data:fcPrices,borderColor:'#10b981',backgroundColor:'#10b98122',fill:true,tension:.3,pointRadius:3,borderWidth:3,borderDash:[]}
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#64748b',maxTicksLimit:15},grid:{color:'#1e293b'}},y:{ticks:{color:'#64748b',callback:v=>'₹'+v},grid:{color:'#1e293b'}}}}});
  }catch(e){document.getElementById('predResult').innerHTML=`<div class="alert warn">⚠️ ${e.message}</div>`}
  finally{btn.disabled=false;btn.innerHTML='🚀 Generate 14-Day Forecast'}
}
</script></body></html>'''

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _j(self,d,c=200):
        b=json.dumps(d).encode()
        self.send_response(c);self.send_header('Content-Type','application/json');self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path in ('/','index.html'):
            page = HTML.replace('EVAL_DATA',json.dumps(EVAL)).replace('MANDI_COMS',json.dumps(mandi_commodities))
            b=page.encode();self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
        else: self.send_error(404)
    def do_POST(self):
        body=self.rfile.read(int(self.headers.get('Content-Length',0)))
        if self.path=='/api/predict':
            d=json.loads(body)
            result=predict_pair(d['mandi'],d['commodity'])
            if result is None: self._j({"error":"Not enough data for this pair (need 44+ days)"},400)
            else: self._j(result)
        else: self.send_error(404)

if __name__=='__main__':
    PORT=8888
    srv=http.server.HTTPServer(('0.0.0.0',PORT),H)
    print(f"\n{'='*50}\n  TFT Dashboard: http://localhost:{PORT}\n{'='*50}\n")
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped."); srv.server_close()
