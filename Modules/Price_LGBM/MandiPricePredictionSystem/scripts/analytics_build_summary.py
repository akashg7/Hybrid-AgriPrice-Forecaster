# scripts/analytics_build_summary.py

from pathlib import Path
import numpy as np
import pandas as pd

FORECAST_PATH = Path("outputs/mandi_commodity_all_forecasts_moirai.csv")
OUT_SUMMARY_PATH = Path("outputs/analytics_mandi_commodity_summary.csv")


def compute_group_stats(group: pd.DataFrame) -> pd.Series:
    """
    Compute summary stats for one (Mandi, Commodity) group
    over the Moirai forecast horizon.
    """
    prices = group["pred_modal_price"].values.astype(float)

    mean_price = float(np.mean(prices))
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    std_price = float(np.std(prices))

    # volatility as % of mean (avoid divide by zero)
    vol_pct = float(std_price / mean_price) if mean_price != 0 else np.nan

    # simple linear trend: price ~ a * t + b
    t = np.arange(len(prices))
    if len(prices) >= 2:
        slope, intercept = np.polyfit(t, prices, 1)
    else:
        slope, intercept = np.nan, np.nan

    # day-to-day percentage changes
    if len(prices) >= 2:
        pct_changes = np.diff(prices) / prices[:-1]
        max_up = float(np.max(pct_changes))
        max_down = float(np.min(pct_changes))
        max_abs_move = float(np.max(np.abs(pct_changes)))
    else:
        max_up = max_down = max_abs_move = np.nan

    # spike / crash flags (thresholds can be tuned)
    spike_flag = bool(max_up > 0.10)      # > +10% jump in one day
    crash_flag = bool(max_down < -0.10)   # < -10% drop in one day

    return pd.Series(
        {
            "forecast_mean_price": mean_price,
            "forecast_min_price": min_price,
            "forecast_max_price": max_price,
            "forecast_std_price": std_price,
            "forecast_volatility_pct": vol_pct,
            "trend_slope_per_day": float(slope),
            "max_daily_pct_up": max_up,
            "max_daily_pct_down": max_down,
            "max_abs_daily_pct_move": max_abs_move,
            "has_spike": spike_flag,
            "has_crash": crash_flag,
        }
    )


def main():
    print(f"Loading Moirai forecasts from {FORECAST_PATH} ...")
    df = pd.read_csv(FORECAST_PATH)

    # ensure date is datetime for potential future use
    df["date"] = pd.to_datetime(df["date"])

    group_cols = [c for c in ["Mandi", "Commodity", "State", "district_name"] if c in df.columns]

    print(f"Grouping by: {group_cols}")
    grouped = df.groupby(group_cols, dropna=False)

    summary_rows = grouped.apply(compute_group_stats).reset_index()

    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_rows.to_csv(OUT_SUMMARY_PATH, index=False)

    print(f"Saved mandi+commodity summary analytics to: {OUT_SUMMARY_PATH}")
    print(f"Summary shape: {summary_rows.shape}")


if __name__ == "__main__":
    main()
