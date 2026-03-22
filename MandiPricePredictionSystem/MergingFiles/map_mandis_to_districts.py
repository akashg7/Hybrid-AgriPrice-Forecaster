import pandas as pd
from rapidfuzz import process, fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

print("\n=== STEP 1: LOAD FILES ===")

full_df = pd.read_csv("ALL_CROPS_DATA.csv")
official = pd.read_csv("agmarknet_master_parallel.csv")

print("Full data:", full_df.shape)
print("Official mandis:", official.shape)

# Clean columns
full_df["market_clean"] = (
    full_df["Mandi"]
    .str.upper()
    .str.replace(r"[^A-Z ]", "", regex=True)
    .str.strip()
)

official["market_clean"] = (
    official["market_name"]
    .str.upper()
    .str.replace(r"[^A-Z ]", "", regex=True)
    .str.strip()
)

# Unique mandis (ONLY 2795 rows)
mandis = full_df[["Mandi", "State", "market_clean"]].drop_duplicates().reset_index(drop=True)
print("\nUnique mandis:", mandis.shape)

# Precompute choices array for fuzzy search
choices = official["market_clean"].tolist()

###############################################################
# PARALLEL MATCHING
###############################################################

def match_mandi(row):
    mc = row["market_clean"]

    # Exact match first
    exact_match = official[official["market_clean"] == mc]
    if len(exact_match) > 0:
        off = exact_match.iloc[0]
        return {
            "Mandi": row["Mandi"],
            "State": row["State"],
            "district_name": off["district_name"],
            "district_id": off["district_id"],
            "state_id": off["state_id"],
        }

    # Fuzzy match
    best = process.extractOne(mc, choices, scorer=fuzz.token_sort_ratio)
    off = official[official["market_clean"] == best[0]].iloc[0]
    return {
        "Mandi": row["Mandi"],
        "State": row["State"],
        "district_name": off["district_name"],
        "district_id": off["district_id"],
        "state_id": off["state_id"],
    }

print("\n=== STEP 2: PARALLEL MATCHING (FAST) ===")

results = []
with ThreadPoolExecutor(max_workers=32) as ex:
    futures = {ex.submit(match_mandi, row): idx for idx, row in mandis.iterrows()}
    for fut in as_completed(futures):
        results.append(fut.result())

mapped_unique = pd.DataFrame(results)
mapped_unique.to_csv("mapped_unique_mandis.csv", index=False)

print("Saved mapped_unique_mandis.csv:", mapped_unique.shape)

###############################################################
# MERGE BACK INTO FULL DATA (VECTORIZED, FAST)
###############################################################

full_with_district = full_df.merge(
    mapped_unique[["Mandi", "State", "district_name", "state_id"]],
    on=["Mandi", "State"],
    how="left"
)

full_with_district.to_csv("full_with_district.csv", index=False)
print("Saved full_with_district.csv:", full_with_district.shape)

###############################################################
# UNIQUE DISTRICTS (SMALL)
###############################################################

districts = mapped_unique[["district_name", "state_id"]].drop_duplicates()
districts.to_csv("districts_unique.csv", index=False)

print("Saved districts_unique.csv:", districts.shape)
