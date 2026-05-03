# scripts/moirai_forecast_single.py
import sys
sys.path.append("/Users/akashg/Desktop/uni2ts/src")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ---------- CONFIG ----------
DATA_PATH = Path("data/processed/mandi_moirai_scaled.parquet")
TARGET_SCALER_PATH = Path("models/moirai_target_scaler.pkl")

PRED_LEN = 14        # forecast horizon
CTX_LEN = 200        # how much context the model sees
PATCH_SIZE = "auto"  # let Moirai choose
NUM_SAMPLES = 100    # MC samples for probabilistic forecast
FREQ = "D"           # daily data
BATCH_SIZE = 32


# ---------- DATA HELPERS ----------

def load_scaled_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_series_df(df: pd.DataFrame, series_id) -> pd.DataFrame:
    """
    Filter the scaled dataframe to one series_id and return
    a DataFrame with columns: ['date', 'y'] sorted by date.
    """
    sub = df[df["series_id"] == series_id].copy()
    if sub.empty:
        raise ValueError(f"No rows found for series_id={series_id}")

    sub = sub.sort_values("date")
    return sub[["date", "y"]]


def build_listdataset_from_series(series_df: pd.DataFrame) -> ListDataset:
    """
    Convert a single-series DataFrame (date, y) to a GluonTS ListDataset.
    Moirai via Uni2TS works nicely with this.
    """
    start = series_df["date"].iloc[0]
    target = series_df["y"].values.astype("float32")

    dataset = ListDataset(
        [
            {
                "start": start,
                "target": target,
            }
        ],
        freq=FREQ,
    )
    return dataset


# ---------- MOIRAI MODEL LOADING ----------

def load_moirai_predictor():
    """
    Load pre-trained Moirai 1.1 R-small via Uni2TS in zero-shot mode.
    """
    print("Loading Moirai 1.1 R-small from Hugging Face ...")
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

    model = MoiraiForecast(
        module=module,
        prediction_length=PRED_LEN,
        context_length=CTX_LEN,
        patch_size=PATCH_SIZE,
        num_samples=NUM_SAMPLES,
        target_dim=1,                 # univariate target (y)
        feat_dynamic_real_dim=0,      # no extra dynamic covariates (for now)
        past_feat_dynamic_real_dim=0, # no past dynamic covariates
    )

    predictor = model.create_predictor(batch_size=BATCH_SIZE)
    return predictor


# ---------- FORECAST FUNCTION ----------

def forecast_series(series_id):
    """
    Run Moirai forecast for a single series_id, return a DataFrame with:
    [date, series_id, pred_modal_price]
    """
    print(f"Loading scaled data from {DATA_PATH} ...")
    df = load_scaled_data()

    print(f"Building series for series_id={series_id} ...")
    series_df = build_series_df(df, series_id)

    if len(series_df) < PRED_LEN + 10:
        raise ValueError(
            f"Series {series_id} too short for forecasting (len={len(series_df)})"
        )

    dataset = build_listdataset_from_series(series_df)

    print("Loading Moirai predictor ...")
    predictor = load_moirai_predictor()

    print("Running forecast ...")
    forecasts = list(predictor.predict(dataset))

    if len(forecasts) == 0:
        raise RuntimeError("No forecasts returned by Moirai.")

    forecast = forecasts[0]

    # forecast.samples could be:
    # - (num_samples, prediction_length)
    # - (num_samples, prediction_length, target_dim)
    samples = forecast.samples  # numpy array

    mean_scaled = samples.mean(axis=0)  # avg over samples

    if mean_scaled.ndim == 2:
        mean_scaled = mean_scaled[:, 0]
    elif mean_scaled.ndim == 1:
        pass
    else:
        raise RuntimeError(f"Unexpected samples shape: {samples.shape}")

    # Build future dates...
    last_date = series_df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=PRED_LEN,
        freq=FREQ,
    )

    target_scaler = joblib.load(TARGET_SCALER_PATH)
    mean_scaled_2d = mean_scaled.reshape(-1, 1)
    mean_inv = target_scaler.inverse_transform(mean_scaled_2d).ravel()

    out_df = pd.DataFrame(
        {
            "date": future_dates,
            "series_id": series_id,
            "pred_modal_price": mean_inv,
        }
    )


    return out_df

def forecast_series_with_predictor(
    predictor,
    df_scaled: pd.DataFrame,
    series_id,
    pred_len: int = PRED_LEN,
):
    """
    Same as forecast_series(), but reuses an existing predictor and df.
    """
    print(f"[Moirai] Forecasting series_id={series_id} ...")

    series_df = df_scaled[df_scaled["series_id"] == series_id].copy()
    if series_df.empty:
        print(f"[SKIP] No data for series_id={series_id}")
        return None

    series_df = series_df.sort_values("date")
    if len(series_df) < pred_len + 10:
        print(f"[SKIP] Too short series_id={series_id} (len={len(series_df)})")
        return None

    series_df = series_df[["date", "y"]]
    dataset = build_listdataset_from_series(series_df)

    forecasts = list(predictor.predict(dataset))
    if not forecasts:
        print(f"[SKIP] No forecast for series_id={series_id}")
        return None

    forecast = forecasts[0]
    samples = forecast.samples  # numpy array

    mean_scaled = samples.mean(axis=0)
    if mean_scaled.ndim == 2:
        mean_scaled = mean_scaled[:, 0]
    elif mean_scaled.ndim == 1:
        pass
    else:
        raise RuntimeError(f"Unexpected samples shape: {samples.shape}")

    last_date = series_df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=pred_len,
        freq=FREQ,
    )

    target_scaler = joblib.load(TARGET_SCALER_PATH)
    mean_scaled_2d = mean_scaled.reshape(-1, 1)
    mean_inv = target_scaler.inverse_transform(mean_scaled_2d).ravel()

    out_df = pd.DataFrame(
        {
            "date": future_dates,
            "series_id": series_id,
            "pred_modal_price": mean_inv,
        }
    )
    return out_df


# ---------- DEMO MAIN ----------

def demo():
    """
    Demo: pick the first series_id with enough length and forecast it.
    Saves CSV to outputs/moirai_demo_forecast.csv
    """
    df = load_scaled_data()
    counts = df.groupby("series_id").size().reset_index(name="n")
    # pick series with length >= 300 as safe
    long_series = counts[counts["n"] >= 300]

    if long_series.empty:
        raise RuntimeError("No series_id has length >= 300")

    row = long_series.iloc[0]
    sid = row["series_id"]
    print(f"Demo using series_id={sid} (n={row['n']})")

    forecast_df = forecast_series(sid)
    out_path = Path("outputs/moirai_demo_forecast.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(out_path, index=False)
    print(f"Saved demo Moirai forecast to {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # user passed series_id explicitly
        series_id_arg = sys.argv[1]
        try:
            # try cast to int, fallback to string
            series_id_arg = int(series_id_arg)
        except ValueError:
            pass

        df_out = forecast_series(series_id_arg)
        print(df_out)
    else:
        demo()
