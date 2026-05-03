import os
import json
import warnings
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import Modular Engines
from Engines.TFT_Engine import TFTEngine
from Engines.LGBM_Engine import LGBMEngine
from Engines.Crop_Engine import CropEngine
from Engines.Disease_Engine import DiseaseEngine

warnings.filterwarnings("ignore")

app = FastAPI(title="AgriSense Unified Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HUB CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MOD_TFT = BASE_DIR / "Modules/Price_TFT"
MOD_LGBM = BASE_DIR / "Modules/Price_LGBM"
MOD_CROP = BASE_DIR / "Modules/Crop_Rec"

# Registry for Engines
hub = {
    "tft": None,
    "lgbm": None,
    "crop": None,
    "disease": None,
    "hierarchy": {}
}

def initialize_hub():
    print("🚀 Booting AgriSense Modular Hub...")
    
    # 1. Initialize TFT Engine
    hub["tft"] = TFTEngine(
        model_path=MOD_TFT / "models/tft-heavy-model/best-advanced-tft-epoch=15-val_loss=44.81.ckpt",
        features_path=MOD_TFT / "data/features.csv"
    )

    # 2. Initialize LGBM Engine
    hub["lgbm"] = LGBMEngine(
        model_path=MOD_LGBM / "price_lgbm_full_features.pkl",
        features_path=MOD_TFT / "data/features.csv" # Uses the same base features
    )

    # 3. Initialize Crop Engine
    hub["crop"] = CropEngine(
        model_path=MOD_CROP / "models/crop_model_ui.pkl"
    )

    # 4. Initialize Disease Engine
    hub["disease"] = DiseaseEngine()

    # 5. Build Hierarchy from the corrected map
    map_df = pd.read_csv(MOD_TFT / "data/hierarchy_map.csv")
    h = {}
    
    # Calculate valid pairs (Min 74 rows for 60-enc / 14-dec)
    counts = hub["tft"].df.groupby(["Mandi", "Commodity"]).size()
    valid_pairs = set(counts[counts >= 74].index)
    
    for _, row in map_df.iterrows():
        s, d, m = str(row["state_name"]), str(row["district_name"]), str(row["market_name"])
        # Check all commodities for this mandi
        if m in hub["tft"].df["Mandi"].values:
            m_coms = hub["tft"].df[hub["tft"].df["Mandi"] == m]["Commodity"].unique()
            has_valid = any((m, c) in valid_pairs for c in m_coms)
            
            if has_valid:
                if s not in h: h[s] = {}
                if d not in h[s]: h[s][d] = []
                if m not in h[s][d]: h[s][d].append(m)
    
    hub["hierarchy"] = {s: {d: sorted(ms) for d, ms in ds.items() if ms} for s, ds in h.items() if ds}
    print("✅ All Engines Synchronized.")

@app.get("/api/hierarchy")
async def get_hierarchy():
    return JSONResponse(hub["hierarchy"])

@app.get("/api/commodities/{mandi}")
async def get_commodities(mandi: str):
    m_df = hub["tft"].df[hub["tft"].df["Mandi"] == mandi]
    counts = m_df.groupby("Commodity").size()
    valid_coms = counts[counts >= 74].index.tolist()
    return JSONResponse(sorted(valid_coms))

@app.post("/api/tft/predict")
async def tft_predict(request: Request):
    data = await request.json()
    res = hub["tft"].predict(data.get("mandi"), data.get("commodity"))
    return JSONResponse(res if res else {"error": "Insufficient data"})

@app.post("/api/lgbm/predict")
async def lgbm_predict(request: Request):
    data = await request.json()
    res = hub["lgbm"].predict(data.get("mandi"), data.get("commodity"))
    return JSONResponse(res if res else {"error": "Insufficient data"})

@app.post("/api/crop/recommend")
async def crop_recommend(request: Request):
    data = await request.json()
    recs = hub["crop"].recommend(data)
    return JSONResponse({"recommendations": recs})

@app.post("/api/disease/detect")
async def disease_detect(file: UploadFile = File(...)):
    res = hub["disease"].detect_v2(file)
    return JSONResponse(res)

if __name__ == "__main__":
    initialize_hub()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
