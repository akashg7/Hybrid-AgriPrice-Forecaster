# scripts/query_price_forecast.py

from pathlib import Path
import pandas as pd

FORECAST_PATH = Path("outputs/mandi_commodity_all_forecasts_moirai.csv")


def get_forecast(mandi: str, commodity: str) -> pd.DataFrame:
    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[(df["Mandi"] == mandi) & (df["Commodity"] == commodity)].copy()
    return sub.sort_values("date")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mandi", required=True, help="Exact Mandi name")
    parser.add_argument("--commodity", required=True, help="Exact Commodity name")
    args = parser.parse_args()

    sub = get_forecast(args.mandi, args.commodity)

    if sub.empty:
        print("No forecast found for this mandi+commodity.")
    else:
        print(sub[["date", "pred_modal_price"]])


if __name__ == "__main__":
    main()
