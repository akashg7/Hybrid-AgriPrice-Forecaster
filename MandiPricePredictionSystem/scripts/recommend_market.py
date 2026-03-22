# scripts/recommend_market.py

import pandas as pd
from pathlib import Path

SUMMARY_PATH = Path("outputs/analytics_mandi_commodity_summary.csv")


def recommend_markets(
    commodity: str,
    state: str | None = None,
    district: str | None = None,
    top_k: int = 10,
    risk_aversion: float = 0.0,  # 0 = ignore volatility, >0 penalize
):
    s = pd.read_csv(SUMMARY_PATH)

    sub = s[s["Commodity"] == commodity].copy()

    if state is not None:
        sub = sub[sub["State"] == state]

    if district is not None and "district_name" in sub.columns:
        sub = sub[sub["district_name"] == district]

    if sub.empty:
        return sub

    # risk-adjusted score = mean_price - risk_aversion * volatility_pct * mean_price
    sub["risk_adjusted_score"] = (
        sub["forecast_mean_price"] * (1.0 - risk_aversion * sub["forecast_volatility_pct"])
    )

    sub = sub.sort_values("risk_adjusted_score", ascending=False)

    return sub.head(top_k)[
        ["Mandi", "Commodity", "State", "district_name",
         "forecast_mean_price", "forecast_volatility_pct",
         "trend_slope_per_day", "has_spike", "has_crash", "risk_adjusted_score"]
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", required=True)
    parser.add_argument("--state", default=None)
    parser.add_argument("--district", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--risk_aversion", type=float, default=0.0)
    args = parser.parse_args()

    recs = recommend_markets(
        commodity=args.commodity,
        state=args.state,
        district=args.district,
        top_k=args.top_k,
        risk_aversion=args.risk_aversion,
    )

    if recs.empty:
        print("No markets found for these filters.")
    else:
        print(recs)


if __name__ == "__main__":
    main()
