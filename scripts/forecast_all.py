# scripts/forecast_all.py
from pathlib import Path
import time
import pandas as pd

from forecast_single import (
    load_full_data,
    load_model,
    ml_forecast_single_pair,
    save_forecast,
    PAIRS_PATH,
    OUT_DIR,
    FORECAST_HORIZON,
)

MAX_PAIRS = 200  # start with 200, later you can remove this limit


def main():
    start_time = time.time()

    print("Loading full data...")
    full_df = load_full_data()

    print("Loading mandi+commodity pairs...")
    pairs_df = pd.read_csv(PAIRS_PATH)

    n_pairs_total = len(pairs_df)
    print(f"Total pairs available: {n_pairs_total}")
    print(f"Will process up to MAX_PAIRS={MAX_PAIRS}")

    print("Loading model...")
    model = load_model()

    success = 0
    skipped = 0
    errors = 0

    for i, row in pairs_df.iterrows():
        if i >= MAX_PAIRS:
            print(f"Reached MAX_PAIRS={MAX_PAIRS}, stopping early.")
            break

        mandi_name = row["Mandi"]
        commodity_name = row["Commodity"]

        try:
            forecast_df = ml_forecast_single_pair(
                model,
                full_df,
                mandi_name=mandi_name,
                commodity_name=commodity_name,
                horizon=FORECAST_HORIZON,
            )

            if forecast_df is None or forecast_df.empty:
                skipped += 1
            else:
                save_forecast(forecast_df, mandi_name, commodity_name)
                success += 1

        except Exception as e:
            print(f"[ERROR] {mandi_name} - {commodity_name}: {e}")
            errors += 1

        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(
                f"Progress: {i+1}/{min(n_pairs_total, MAX_PAIRS)} | "
                f"success={success}, skipped={skipped}, errors={errors}, "
                f"elapsed={elapsed:.1f} min"
            )

    total_time = (time.time() - start_time) / 60
    print(
        "\nFinished ML forecasting.\n"
        f"Success: {success}, Skipped: {skipped}, Errors: {errors}, "
        f"Total time: {total_time:.1f} minutes"
    )

    # Optional: combine forecasts from this run into one CSV
    combine_all = True
    if combine_all:
        combine_all_forecasts()


def combine_all_forecasts():
    """Combine all forecast.csv files under OUT_DIR into a single CSV."""
    print("\nCombining all forecast files into a single CSV...")
    files = list(OUT_DIR.rglob("forecast.csv"))

    if not files:
        print("No forecast.csv files found. Skipping combination.")
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    all_forecasts = pd.concat(dfs, ignore_index=True)

    out_path = Path("outputs/mandi_commodity_all_forecasts_ml.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_forecasts.to_csv(out_path, index=False)

    print(f"Combined ML forecast saved to: {out_path}")


if __name__ == "__main__":
    main()
