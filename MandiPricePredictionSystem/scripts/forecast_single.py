# scripts/forecast_single.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

# ---- CONFIG ----
DATA_PATH = Path("merged_crop_data_with_weather.csv")
PAIRS_PATH = Path("mandi_crop_pairs.csv")
OUT_DIR = Path("outputs/forecasts")
FORECAST_HORIZON = 14
MODEL_PATH = Path("models/price_lgbm_model.pkl")


def load_full_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)
    return model


def get_series_for_pair(full_df: pd.DataFrame, mandi_name: str, commodity_name: str) -> pd.DataFrame:
    mask = (full_df["Mandi"] == mandi_name) & (full_df["Commodity"] == commodity_name)
    sub = full_df.loc[mask].copy()
    sub = sub.sort_values("date")
    return sub


def _sanitize_for_path(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def save_forecast(forecast_df: pd.DataFrame, mandi_name: str, commodity_name: str) -> Path:
    safe_mandi = _sanitize_for_path(mandi_name)
    safe_commodity = _sanitize_for_path(commodity_name)

    out_dir = OUT_DIR / f"Mandi={safe_mandi}" / f"Commodity={safe_commodity}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "forecast.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"[OK] Saved forecast for {mandi_name} - {commodity_name} -> {out_path}")
    return out_path


def make_feature_row_for_date(
    hist: pd.DataFrame,
    forecast_date: pd.Timestamp,
    mandi_name: str,
    commodity_name: str,
) -> pd.DataFrame | None:
    hist = hist.sort_values("date")
    prices = hist["ModalPrice"].values

    if len(prices) < 7:
        return None

    def get_lag(k: int) -> float:
        if len(prices) >= k:
            return float(prices[-k])
        else:
            return np.nan

    lag_1 = get_lag(1)
    lag_2 = get_lag(2)
    lag_3 = get_lag(3)
    lag_7 = get_lag(7)

    if any(np.isnan(v) for v in [lag_1, lag_2, lag_3, lag_7]):
        return None

    dayofweek = forecast_date.dayofweek
    month = forecast_date.month

    data = {
        "lag_1": [lag_1],
        "lag_2": [lag_2],
        "lag_3": [lag_3],
        "lag_7": [lag_7],
        "dayofweek": [dayofweek],
        "month": [month],
        "Mandi": [mandi_name],
        "Commodity": [commodity_name],
    }

    feat_df = pd.DataFrame(data)
    feat_df["Mandi"] = feat_df["Mandi"].astype("category")
    feat_df["Commodity"] = feat_df["Commodity"].astype("category")
    return feat_df


def ml_forecast_single_pair(
    model,
    full_df: pd.DataFrame,
    mandi_name: str,
    commodity_name: str,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame | None:
    series_df = get_series_for_pair(full_df, mandi_name, commodity_name)

    if series_df.empty:
        print(f"[SKIP] No data for {mandi_name} - {commodity_name}")
        return None

    if len(series_df) < 30:
        print(f"[SKIP] Too few rows ({len(series_df)}) for {mandi_name} - {commodity_name}")
        return None

    hist = series_df.sort_values("date")[["date", "ModalPrice"]].copy()

    forecasts = []
    last_date = hist["date"].max()

    for step in range(horizon):
        forecast_date = last_date + pd.Timedelta(days=1)

        feat_row = make_feature_row_for_date(
            hist=hist,
            forecast_date=forecast_date,
            mandi_name=mandi_name,
            commodity_name=commodity_name,
        )

        if feat_row is None:
            print(f"[SKIP] Not enough history for ML forecast {mandi_name} - {commodity_name}")
            return None

        pred_price = float(model.predict(feat_row)[0])

        forecasts.append(
            {
                "date": forecast_date,
                "Mandi": mandi_name,
                "Commodity": commodity_name,
                "pred_modal_price": pred_price,
            }
        )

        hist = pd.concat(
            [hist, pd.DataFrame({"date": [forecast_date], "ModalPrice": [pred_price]})],
            ignore_index=True,
        )
        last_date = forecast_date

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


def main():
    print("Loading full data...")
    full_df = load_full_data()

    print("Loading model...")
    model = load_model()

    # 1) If user passed CLI args → use them
    if len(sys.argv) == 3:
        mandi_name = sys.argv[1]
        commodity_name = sys.argv[2]
        print(f"Using CLI args: {mandi_name} - {commodity_name}")
    else:
        # 2) Else auto-pick first pair with enough history
        print("No CLI args given, auto-selecting a pair with enough history...")
        pairs_df = pd.read_csv(PAIRS_PATH)

        mandi_name = None
        commodity_name = None

        for _, row in pairs_df.iterrows():
            m = row["Mandi"]
            c = row["Commodity"]
            sub = get_series_for_pair(full_df, m, c)
            if len(sub) >= 30:
                mandi_name = m
                commodity_name = c
                print(f"Selected pair: {mandi_name} - {commodity_name} (rows={len(sub)})")
                break

        if mandi_name is None:
            print("Could not find any (Mandi, Commodity) pair with >= 30 rows.")
            return

    print(f"Running ML forecast demo for: {mandi_name} - {commodity_name}")
    forecast_df = ml_forecast_single_pair(model, full_df, mandi_name, commodity_name)
    if forecast_df is None:
        print("Forecast returned None.")
        return

    save_forecast(forecast_df, mandi_name, commodity_name)


if __name__ == "__main__":
    main()
