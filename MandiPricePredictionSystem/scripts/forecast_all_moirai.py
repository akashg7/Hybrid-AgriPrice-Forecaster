# scripts/forecast_all_moirai.py

from pathlib import Path
import time
import pandas as pd

from moirai_forecast_single import (
    load_scaled_data,
    load_moirai_predictor,
    forecast_series_with_predictor,
    PRED_LEN,
)

OUT_DIR = Path("outputs/forecasts_moirai")
MAPPING_PATH = Path("data/metadata/series_mapping.csv")
COMBINED_OUT = Path("outputs/mandi_commodity_all_forecasts_moirai.csv")

# MAX_SERIES = 200  # for safety; later you can remove/raise this
MAX_SERIES = None  # None = process ALL series


def _sanitize_for_path(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def main():
    start = time.time()

    print("Loading scaled data...")
    df_scaled = load_scaled_data()

    print(f"Loading series mapping from {MAPPING_PATH} ...")
    mapping = pd.read_csv(MAPPING_PATH)

    total_series = len(mapping)
    print(f"Total series: {total_series}")
    print(f"Will process up to MAX_SERIES={MAX_SERIES}")

    print("Loading Moirai predictor (once)...")
    predictor = load_moirai_predictor()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    errors = 0
    all_forecasts = []

    # determine denominator for progress display
    progress_den = total_series if MAX_SERIES is None else min(total_series, MAX_SERIES)

    for i, row in mapping.iterrows():
        series_id = int(row["series_id"])

        if MAX_SERIES is not None and series_id >= MAX_SERIES:
            print(f"Reached MAX_SERIES={MAX_SERIES}, stopping early.")
            break

        mandi = row.get("Mandi", None)
        commodity = row.get("Commodity", None)
        state = row.get("State", None)
        district = row.get("district_name", None)

        try:
            forecast_df = forecast_series_with_predictor(
                predictor=predictor,
                df_scaled=df_scaled,
                series_id=series_id,
                pred_len=PRED_LEN,
            )

            if forecast_df is None or forecast_df.empty:
                skipped += 1
                continue

            # Add metadata columns
            forecast_df["Mandi"] = mandi
            forecast_df["Commodity"] = commodity
            forecast_df["State"] = state
            forecast_df["district_name"] = district

            all_forecasts.append(forecast_df)

            # save per-mandi+commodity file
            safe_mandi = _sanitize_for_path(mandi) if mandi is not None else f"series_{series_id}"
            safe_commodity = _sanitize_for_path(commodity) if commodity is not None else "Unknown"

            out_dir = OUT_DIR / f"Mandi={safe_mandi}" / f"Commodity={safe_commodity}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "forecast.csv"
            forecast_df.to_csv(out_path, index=False)

            print(f"[OK] Saved {len(forecast_df)} rows for {mandi} - {commodity} -> {out_path}")
            success += 1

        except Exception as e:
            print(f"[ERROR] series_id={series_id}, {mandi} - {commodity}: {e}")
            errors += 1

        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start) / 60
            print(
                f"Progress: {i+1}/{progress_den} | "
                f"success={success}, skipped={skipped}, errors={errors}, "
                f"elapsed={elapsed:.1f} min"
            )

    total_time = (time.time() - start) / 60
    print(
        "\nFinished Moirai forecasting.\n"
        f"Success: {success}, Skipped: {skipped}, Errors: {errors}, "
        f"Total time: {total_time:.1f} minutes"
    )

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(COMBINED_OUT, index=False)
        print(f"Combined Moirai forecasts saved to: {COMBINED_OUT}")
    else:
        print("No forecasts generated, skipping combined CSV.")


if __name__ == "__main__":
    main()
