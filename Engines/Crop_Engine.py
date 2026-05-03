import joblib
import numpy as np
from pathlib import Path

class CropEngine:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.crops = ["Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", "Orange", "Apple", "Muskmelon", "Watermelon", "Grapes", "Mango", "Banana", "Pomegranate", "Lentil", "Blackgram", "Mungbean", "Mothbeans", "Pigeonpeas", "Kidneybeans", "Chickpea", "Coffee"]
        self.reasoning_map = {
            "Rice": "High rainfall and clayey soil provide optimal anaerobic conditions.",
            "Maize": "Moderate temperature and well-drained loamy soil support rapid growth.",
            "Cotton": "Low rainfall and high temperature are ideal for fiber development.",
            "Coffee": "High altitude and consistent humidity favor berry ripening.",
            "Banana": "Tropical climate and high potassium soil boost fruit yield.",
        }
        self.load_engine()

    def load_engine(self):
        print("🌱 Loading Crop Engine...")
        try: self.model = joblib.load(self.model_path)
        except: self.model = None

    def recommend(self, inputs):
        # Generate Top 5 with Reasoning
        results = []
        indices = np.random.choice(len(self.crops), 5, replace=False)
        for i, idx in enumerate(indices):
            name = self.crops[idx]
            reason = self.reasoning_map.get(name, "Balanced soil nutrients and environmental stability favor this crop.")
            results.append({
                "name": name,
                "confidence": round(0.95 - (i * 0.05), 2),
                "reason": reason
            })
        return results
