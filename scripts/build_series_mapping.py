# scripts/build_series_mapping.py

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/mandi_moirai_scaled.parquet")
OUT_PATH = Path("data/metadata/series_mapping.csv")


def main():
    print(f"Loading scaled data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # Keep one representative row per series_id
    cols = ["series_id", "Mandi", "Commodity", "State", "district_name"]
    cols = [c for c in cols if c in df.columns]

    mapping = (
        df[cols]
        .drop_duplicates(subset=["series_id"])
        .sort_values("series_id")
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUT_PATH, index=False)

    print(f"Saved series mapping ({len(mapping)} rows) to {OUT_PATH}")


if __name__ == "__main__":
    main()
