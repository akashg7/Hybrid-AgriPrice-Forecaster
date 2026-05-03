import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz import process, fuzz


###############################################################
# STEP 1: LOAD FULL DATA & EXTRACT UNIQUE MANDIS (NOT 2.7M rows)
###############################################################

full_df = pd.read_csv("ALL_CROPS_DATA.csv")  
# must contain columns 'Mandi', 'State', 'Date'

print("Loaded full data:", full_df.shape)

# Extract only ~2795 unique mandis
your_df = full_df[["Mandi", "State"]].drop_duplicates().reset_index(drop=True)
print("Unique mandis:", your_df.shape)

# Clean text for matching
your_df["market_clean"] = (
    your_df["Mandi"]
    .str.upper()
    .str.replace(r"[^A-Z ]", "", regex=True)
    .str.strip()
)

your_df["state_clean"] = your_df["State"].str.upper().str.strip()


###############################################################
# STEP 2: LOAD OFFICIAL API MANDI MASTER
###############################################################

official = pd.read_csv("agmarknet_master_parallel.csv")
# columns: state_id, state_name, district_id, district_name, market_id, market_name

official["market_clean"] = (
    official["market_name"]
    .str.upper()
    .str.replace(r"[^A-Z ]", "", regex=True)
    .str.strip()
)

print("Loaded official:", official.shape)



###############################################################
# STEP 3: FAST EXACT + FUZZY MATCH (ONLY ON UNIQUE MANDIS)
###############################################################

matches = []
choices = official["market_clean"].tolist()

print("\nMatching unique mandis to official list...")

for _, row in your_df.iterrows():
    name = row["market_clean"]

    # Exact match first
    exact = official[official["market_clean"] == name]
    if len(exact) > 0:
        off = exact.iloc[0]
        score = 100
    else:
        # Fuzzy match fallback
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        off = official[official["market_clean"] == best[0]].iloc[0]
        score = best[1]

    matches.append({
        "Mandi": row["Mandi"],
        "State": row["State"],
        "market_id": off["market_id"],
        "matched_market": off["market_name"],
        "district_id": off["district_id"],
        "district_name": off["district_name"],
        "state_id": off["state_id"],
        "confidence": score
    })

mapped_df = pd.DataFrame(matches)
mapped_df.to_csv("mapped_unique_mandis.csv", index=False)
print("Saved unique mapping:", mapped_df.shape)



###############################################################
# STEP 4: GET UNIQUE DISTRICTS FOR WEATHER (Only 200–400 rows)
###############################################################

districts = mapped_df[["district_name", "state_id"]].drop_duplicates().reset_index(drop=True)
print("Unique districts:", districts.shape)


###############################################################
# DISTRICT NAME NORMALIZATION (for better geocoding)
###############################################################

def clean_district_name(name: str) -> str:
    """Normalize district name for geocoding."""
    if pd.isna(name):
        return ""
    n = name.upper().strip()
    # Remove common junk
    replacements = [
        ("(BARODA)", ""),
        ("(VADODARA)", ""),
        ("(SUBJI MANDI)", ""),
        (" DISTRICT", ""),
        ("-", " "),
        ("_", " "),
    ]
    for old, new in replacements:
        n = n.replace(old, new)
    n = " ".join(n.split())  # collapse multiple spaces
    return n


# manual alias corrections for known problematic spellings
ALIAS = {
    "HISSAR": "HISAR",
    "KAITHAR": "KATIHAR",
    "KACHCHH": "KUTCH",
    "MEHSANA": "MAHESANA",
    "VADODARA(BARODA)": "VADODARA",
    "NORTH GOA": "NORTH GOA",
    "SOUTH GOA": "SOUTH GOA",
}

def district_query_name(name: str) -> str:
    c = clean_district_name(name)
    return ALIAS.get(c, c)


###############################################################
# STEP 5: PARALLEL GEOCODING
###############################################################

def geocode(name):
    query = district_query_name(name)
    if not query:
        print(f"[GEOCODE SKIP] Empty name for {name}")
        return None, None

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}, India&count=1"
    try:
        r = requests.get(url, timeout=10)
        js = r.json()
        if "results" in js and len(js["results"]) > 0:
            lat = js["results"][0]["latitude"]
            lon = js["results"][0]["longitude"]
            return lat, lon
        else:
            print(f"[GEOCODE MISS] {name} → query='{query}' returned no results")
    except Exception as e:
        print(f"[GEOCODE ERR] {name} → {e}")
    return None, None

districts["lat"] = None
districts["lon"] = None

print("\nGeocoding districts...")

with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {
        ex.submit(geocode, row["district_name"]): idx
        for idx, row in districts.iterrows()
    }

    for fut in as_completed(futures):
        idx = futures[fut]
        lat, lon = fut.result()
        districts.loc[idx, "lat"] = lat
        districts.loc[idx, "lon"] = lon
        print(f"Geocoded {idx + 1}/{len(districts)} → {districts.loc[idx, 'district_name']} ({lat}, {lon})")

districts.to_csv("district_geocoded.csv", index=False)



###############################################################
# STEP 6: PARALLEL WEATHER FETCH (SAFE)
###############################################################

START = "2024-01-01"
END   = "2025-12-01"

def fetch_weather(row):
    dist = row["district_name"]
    lat = row["lat"]
    lon = row["lon"]

    # Skip if missing coordinates
    if pd.isna(lat) or pd.isna(lon):
        print(f"[SKIP] No coordinates for {dist}")
        return []

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={START}&end_date={END}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=auto"
    )

    # ---------------- SAFE API REQUEST ----------------
    try:
        r = requests.get(url, timeout=20)
        js = r.json()
    except Exception as e:
        print(f"[API ERROR] {dist}: {e}")
        return []

    # ---------------- KEY FIX: CHECK 'daily' EXISTS ----------------
    if "daily" not in js or js["daily"] is None:
        print(f"[NO WEATHER] {dist} → No daily section in API response")
        return []

    daily = js["daily"]

    # Handle empty-day response (rare, but happens)
    if "time" not in daily or len(daily["time"]) == 0:
        print(f"[EMPTY] {dist} → No daily time data")
        return []

    days = daily["time"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    rain = daily["precipitation_sum"]

    # ---------------- BUILD OUTPUT ROWS SAFELY ----------------
    out = []
    for i in range(len(days)):
        out.append({
            "district_name": dist,
            "state_id": row["state_id"],
            "date": days[i],
            "tmax": tmax[i],
            "tmin": tmin[i],
            "rain": rain[i],
        })

    return out



weather_rows = []

print("\nFetching weather in parallel...")

with ThreadPoolExecutor(max_workers=25) as ex:
    futures = {
        ex.submit(fetch_weather, row): row["district_name"]
        for _, row in districts.iterrows()
    }

    for fut in as_completed(futures):
        dist = futures[fut]
        rows = fut.result()
        print(f"Weather → {dist}: {len(rows)} rows")
        weather_rows.extend(rows)

df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv("district_weather.csv", index=False)
print("Saved weather:", df_weather.shape)



###############################################################
# STEP 7: MERGE WEATHER BACK TO FULL 2.7M ROWS (FAST)
###############################################################

# Merge mandis → full data
full_with_district = full_df.merge(
    mapped_df[["Mandi", "State", "district_name", "state_id"]],
    on=["Mandi", "State"],
    how="left"
)

# Merge weather using (district, state, date)
full_with_district["Date"] = pd.to_datetime(full_with_district["Date"]).dt.strftime("%Y-%m-%d")

final = full_with_district.merge(
    df_weather,
    left_on=["district_name", "state_id", "Date"],
    right_on=["district_name", "state_id", "date"],
    how="left"
)

final.to_csv("final_full_weather_dataset.csv", index=False)
print("FINAL SHAPE:", final.shape)
print(final.head())
