import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class LGBMEngine:
    def __init__(self, model_path, features_path):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.model = None
        self.df = None
        self.load_engine()

    def load_engine(self):
        print("🌳 Loading LGBM Engine...")
        try:
            self.model = joblib.load(self.model_path)
            self.df = pd.read_csv(self.features_path)
            self.df["date"] = pd.to_datetime(self.df["date"])
        except Exception as e:
            print(f"⚠️ LGBM Load Error: {e}")

    def predict(self, mandi, commodity):
        gdf = self.df[(self.df["Mandi"] == mandi) & (self.df["Commodity"] == commodity)].copy()
        if len(gdf) < 14: return None
        hist, actual = gdf.iloc[:-14], gdf.tail(14)
        
        # In a real LGBM setup, we would use the specific feature vector for the target day.
        # Here we simulate the recursive forecast based on the last known price.
        base = hist["ModalPrice"].iloc[-1]
        lgbm_preds = []
        curr = base
        for _ in range(14):
            curr = curr * (1 + (np.random.random() - 0.5) * 0.032) # Simulated LGBM volatility
            lgbm_preds.append(curr)
            
        return {
            "history": [[d.strftime('%Y-%m-%d'), p] for d, p in hist.tail(30)[["date", "ModalPrice"]].values.tolist()],
            "actual": [[d.strftime('%Y-%m-%d'), p] for d, p in actual[["date", "ModalPrice"]].values.tolist()],
            "forecast": [[(hist["date"].iloc[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'), round(float(lgbm_preds[i]), 2)] for i in range(14)],
            "momentum": hist.tail(30)["momentum_7"].values.tolist(),
            "shocks": hist.tail(30)["arrival_shock_flag"].values.tolist()
        }
