# scripts/list_pairs.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path("merged_crop_data_with_weather.csv")  # adjust path if needed

def main():
    df = pd.read_csv(DATA_PATH)

    print("Columns in file:", df.columns.tolist())

    # Get unique Mandi + Commodity pairs
    pairs = (
        df[["Mandi", "Commodity"]]
        .dropna(subset=["Mandi", "Commodity"])  # just in case there are NaNs
        .drop_duplicates()
        .sort_values(["Mandi", "Commodity"])
        .reset_index(drop=True)
    )

    out_path = Path("mandi_crop_pairs.csv")  # or Path("data/metadata/mandi_crop_pairs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)

    print(f"Saved {len(pairs)} mandi+crop pairs to {out_path}")

if __name__ == "__main__":
    main()
