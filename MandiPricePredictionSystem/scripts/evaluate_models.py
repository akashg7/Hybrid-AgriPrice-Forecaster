# scripts/evaluate_models.py
#
# Evaluate Moirai (zero-shot) vs a naive baseline
# using historical backtesting on multiple series.
#
# Idea:
# - Use the scaled Moirai dataset: data/processed/mandi_moirai_scaled.parquet
# - For each series_id with enough history:
#     * Take last PRED_LEN points as test
#     * Use the rest as context
#     * Naive forecast = last context ModalPrice repeated PRED_LEN times
#     * Moirai forecast = run predictor on truncated df (no leakage), get PRED_LEN preds
# - Compare to actual test ModalPrice using MAE, RMSE, MAPE
#
# NOTE:
# - LightGBM evaluation is left as a TODO stub at the bottom.

from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from moirai_forecast_single import (
    load_scaled_data,
    load_moirai_predictor,
    forecast_series_with_predictor,
    PRED_LEN,
)


DATA_SCALED_PATH = Path("data/processed/mandi_moirai_scaled.parquet")
SERIES_MAPPING_PATH = Path("data/metadata/series_mapping.csv")

# Evaluation config
EVAL_PRED_LEN = PRED_LEN      # use same horizon as production (14)
MIN_SERIES_LEN = 60           # require at least this many points per series
MAX_SERIES_EVAL = 100         # evaluate on at most this many series for speed


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # avoid division by 0
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def select_eval_series(df_scaled: pd.DataFrame) -> List[int]:
    """
    Choose series_id that have enough length
    and return up to MAX_SERIES_EVAL of them.
    """
    counts = df_scaled.groupby("series_id").size()
    valid_series = counts[counts >= (MIN_SERIES_LEN + EVAL_PRED_LEN)].index.tolist()
    valid_series = sorted(valid_series)

    if len(valid_series) > MAX_SERIES_EVAL:
        valid_series = valid_series[:MAX_SERIES_EVAL]

    return valid_series


def evaluate_moirai_vs_naive() -> None:
    print("Loading scaled data...")
    df_scaled = load_scaled_data()  # same as used by Moirai forecasting
    df_scaled["date"] = pd.to_datetime(df_scaled["date"])

    print(f"Total rows in scaled data: {len(df_scaled)}")

    eval_series_ids = select_eval_series(df_scaled)
    print(f"Evaluating on {len(eval_series_ids)} series (series_id).")

    print("Loading Moirai predictor...")
    predictor = load_moirai_predictor()

    metrics_naive: List[Dict] = []
    metrics_moirai: List[Dict] = []

    for idx, sid in enumerate(eval_series_ids):
        df_s_full = df_scaled[df_scaled["series_id"] == sid].sort_values("date").copy()

        if len(df_s_full) < MIN_SERIES_LEN + EVAL_PRED_LEN:
            # extra safety
            continue

        # Split into context and test
        context = df_s_full.iloc[:-EVAL_PRED_LEN]
        test = df_s_full.iloc[-EVAL_PRED_LEN:]

        # Ground truth = actual ModalPrice over last EVAL_PRED_LEN days
        # This column should be present from the original engineered data
        y_true = test["ModalPrice"].to_numpy(dtype=float)

        # --------------------
        # 1) Naive forecast
        # --------------------
        last_price = context["ModalPrice"].iloc[-1]
        y_naive = np.full(shape=EVAL_PRED_LEN, fill_value=float(last_price))

        # --------------------
        # 2) Moirai forecast (backtest style)
        #    Trick: drop test rows for this series from df_scaled so Moirai
        #    only sees context, then forecast from there.
        # --------------------
        ctx_indices = context.index
        df_ctx = df_scaled.drop(index=test.index)

        try:
            forecast_df = forecast_series_with_predictor(
                predictor=predictor,
                df_scaled=df_ctx,
                series_id=sid,
                pred_len=EVAL_PRED_LEN,
            )

            if forecast_df is None or forecast_df.empty:
                print(f"[SKIP] Moirai returned empty for series_id={sid}")
                continue

            y_moirai = forecast_df["pred_modal_price"].to_numpy(dtype=float)

            if len(y_moirai) != EVAL_PRED_LEN:
                print(f"[SKIP] Moirai forecast length mismatch for series_id={sid}")
                continue

        except Exception as e:
            print(f"[ERROR] Moirai forecast failed for series_id={sid}: {e}")
            continue

        # --------------------
        # Compute metrics
        # --------------------
        row_meta = {
            "series_id": sid,
            "n_test": EVAL_PRED_LEN,
        }

        metrics_naive.append(
            {
                **row_meta,
                "mae": mae(y_true, y_naive),
                "rmse": rmse(y_true, y_naive),
                "mape": mape(y_true, y_naive),
            }
        )

        metrics_moirai.append(
            {
                **row_meta,
                "mae": mae(y_true, y_moirai),
                "rmse": rmse(y_true, y_moirai),
                "mape": mape(y_true, y_moirai),
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx+1}/{len(eval_series_ids)} series...")

    # --------------------
    # Aggregate results
    # --------------------
    if not metrics_naive or not metrics_moirai:
        print("No metrics computed; something went wrong or no valid series.")
        return

    df_naive = pd.DataFrame(metrics_naive)
    df_moirai = pd.DataFrame(metrics_moirai)

    print("\n=== Per-series metrics (head) ===")
    print("Naive baseline:")
    print(df_naive.head())
    print("\nMoirai:")
    print(df_moirai.head())

    print("\n=== Aggregate metrics over all evaluated series ===")
    agg_naive = df_naive[["mae", "rmse", "mape"]].mean()
    agg_moirai = df_moirai[["mae", "rmse", "mape"]].mean()

    print("\nNaive baseline (avg over series):")
    print(agg_naive)

    print("\nMoirai (avg over series):")
    print(agg_moirai)

    # save to CSV for reporting
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_naive.to_csv(out_dir / "eval_naive_per_series.csv", index=False)
    df_moirai.to_csv(out_dir / "eval_moirai_per_series.csv", index=False)

    summary_df = pd.DataFrame(
        {
            "model": ["naive", "moirai"],
            "mae": [agg_naive["mae"], agg_moirai["mae"]],
            "rmse": [agg_naive["rmse"], agg_moirai["rmse"]],
            "mape": [agg_naive["mape"], agg_moirai["mape"]],
        }
    )
    summary_df.to_csv(out_dir / "eval_summary.csv", index=False)
    print(f"\nSaved detailed metrics and summary to {out_dir}")


# ------------------------
# TODO: LightGBM evaluation
# ------------------------
#
# Outline (not implemented yet):
#
# def evaluate_lightgbm():
#     - Load data/processed/training_data.parquet
#     - Load models/price_lgbm_model.pkl
#     - Do a time-based split on 'date' (e.g., last 60 days as test)
#     - Build X_test from the same features used in train_model.py
#     - Predict with model.predict(X_test)
#     - Compute MAE, RMSE, MAPE vs target column
#
# Then compare its aggregate metrics with naive and Moirai.


if __name__ == "__main__":
    evaluate_moirai_vs_naive()
