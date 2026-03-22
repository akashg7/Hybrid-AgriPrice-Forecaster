import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from uni2ts.model.moirai.module import MoiraiModule as Moirai

def main():
    print("Loading data...")
    try:
        df = pd.read_csv("merged_crop_data_with_weather.csv")
    except FileNotFoundError:
        print("Error: merged_crop_data_with_weather.csv not found. Please run the gdown command first.")
        return

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    # ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # sort globally (good practice)
    df = df.sort_values("date").reset_index(drop=True)

    print(df.head())
    print(df.dtypes)

    # Time-varying features (observed_cov for Moirai)
    observed_features = [
        "Arrivals",
        "temp_avg","temp_max","temp_min",
        "rainfall","humidity","solar_radiation","wind_speed"
    ]

    # Static features for identity embedding
    static_cols = ["state_id", "district_name", "market_clean", "Commodity", "CropGroup"]

    series_list = []

    print("Processing series...")
    for (mkt, com), g in df.groupby(["market_clean", "Commodity"]):
        g = g.sort_values("date").reset_index(drop=True)
        
        if len(g) < 60:   # skip very small series
            continue
        
        # clean missing values
        g["ModalPrice"] = g["ModalPrice"].ffill()
        g[observed_features] = g[observed_features].ffill().fillna(0)
        
        if g["ModalPrice"].isna().all():
            continue
        
        # static features
        st_id = g["state_id"].iloc[0]
        district = g["district_name"].iloc[0]
        crop_group = g["CropGroup"].iloc[0]
        
        static_feat = np.array([
            float(st_id),
            float(hash(district) % 10000),
            float(hash(mkt) % 10000),
            float(hash(com) % 10000),
            float(hash(crop_group) % 10000),
        ], dtype=np.float32)

        # Build the series dict
        series = {
            "target": g["ModalPrice"].values.astype(np.float32),
            "observed_cov": g[observed_features].values.astype(np.float32),
            "static_feat": static_feat,
            "start": g["date"].iloc[0],
            "meta": {
                "market_clean": mkt,
                "Commodity": com,
                "district_name": district,
                "state_id": int(st_id)
            }
        }
        
        series_list.append(series)

    print(f"Total series processed: {len(series_list)}")
    
    print("Loading Moirai model...")
    model = Moirai.from_pretrained("Salesforce/moirai-1.1-R-small")
    print("Model loaded successfully:")
    print(model)

if __name__ == "__main__":
    main()
