# scripts/analytics_nearby_markets.py

from pathlib import Path
import numpy as np
import pandas as pd

HIST_PATH = Path("data/processed/mandi_feature_engineered.csv")


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km) between two points.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_nearby_markets(df_meta, mandi, max_km=100.0, top_k=10):
    """
    df_meta: one row per mandi with lat/lon
    """
    row = df_meta[df_meta["Mandi"] == mandi]
    if row.empty:
        raise ValueError(f"Mandi {mandi} not found in metadata")

    lat0 = row["lat"].iloc[0]
    lon0 = row["lon"].iloc[0]

    df_meta = df_meta.copy()
    df_meta["distance_km"] = haversine(lat0, lon0, df_meta["lat"], df_meta["lon"])
    df_meta = df_meta[df_meta["Mandi"] != mandi]  # exclude itself
    df_meta = df_meta[df_meta["distance_km"] <= max_km]
    return df_meta.sort_values("distance_km").head(top_k)


def get_last_year_prices(df, mandi, commodity, ref_year):
    """
    Return historical prices for (mandi, commodity) in ref_year-1.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    mask = (
        (df["Mandi"] == mandi)
        & (df["Commodity"] == commodity)
        & (df["year"] == ref_year - 1)
    )
    sub = df[mask].copy()
    return sub.sort_values("date")[["date", "ModalPrice"]]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mandi", required=True)
    parser.add_argument("--commodity", required=True)
    parser.add_argument("--ref_year", type=int, required=True)
    parser.add_argument("--max_km", type=float, default=100.0)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading historical engineered data from {HIST_PATH} ...")
    df = pd.read_csv(HIST_PATH)

    # metadata per mandi
    meta_cols = ["Mandi", "lat", "lon", "State", "district_name"]
    df_meta = df[meta_cols].drop_duplicates(subset=["Mandi"]).dropna(subset=["lat", "lon"])

    neighbors = find_nearby_markets(
        df_meta, mandi=args.mandi, max_km=args.max_km, top_k=args.top_k
    )

    print("\nNearby markets:")
    print(neighbors[["Mandi", "State", "district_name", "distance_km"]])

    print("\nPrevious-year prices for requested mandi+commodity:")
    base = get_last_year_prices(df, args.mandi, args.commodity, args.ref_year)
    print(base.head(20))


if __name__ == "__main__":
    main()
