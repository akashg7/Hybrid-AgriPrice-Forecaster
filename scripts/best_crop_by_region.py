# scripts/best_crop_by_region.py

import pandas as pd
from pathlib import Path

SUMMARY_PATH = Path("outputs/analytics_mandi_commodity_summary.csv")


def best_crops_in_mandi(mandi: str, top_k: int = 10):
    s = pd.read_csv(SUMMARY_PATH)
    sub = s[s["Mandi"] == mandi].copy()
    return (
        sub.sort_values("forecast_mean_price", ascending=False)
        .head(top_k)[
            ["Commodity", "forecast_mean_price",
             "trend_slope_per_day", "forecast_volatility_pct",
             "has_spike", "has_crash"]
        ]
    )


def best_crops_in_state(state: str, top_k: int = 10):
    s = pd.read_csv(SUMMARY_PATH)
    sub = s[s["State"] == state].copy()
    if sub.empty:
        return sub
    grouped = (
        sub.groupby("Commodity")["forecast_mean_price"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )
    return grouped.reset_index(name="avg_forecast_price")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mandi", "state"], required=True)
    parser.add_argument("--name", required=True, help="mandi name or state name")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "mandi":
        res = best_crops_in_mandi(args.name, top_k=args.top_k)
    else:
        res = best_crops_in_state(args.name, top_k=args.top_k)

    if res.empty:
        print("No data found.")
    else:
        print(res)


if __name__ == "__main__":
    main()
