import requests
import pandas as pd
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from tqdm import tqdm

# ---------------------------------------
# LOGGING SETUP
# ---------------------------------------
logging.basicConfig(
    filename="scrape.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log(msg):
    print(msg)
    logging.info(msg)


# ---------------------------------------
# API CONFIG
# ---------------------------------------
URL = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/commodity-market/daily-report-weighted"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
}

CROP_IDS = [23, 65, 24, 1, 3, 26, 19, 10, 60, 73]

OUTPUT_FILE = "ALL_CROPS_DATA.csv"
PROGRESS_FILE = "progressTracker1.txt"


# ---------------------------------------
# LOAD LAST PROGRESS
# ---------------------------------------
def load_progress(default_start):
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            date_str = f.read().strip()
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except:
                pass
    return default_start


# ---------------------------------------
# SAVE PROGRESS
# ---------------------------------------
def save_progress(date_str):
    with open(PROGRESS_FILE, "w") as f:
        f.write(date_str)


# ---------------------------------------
# FETCH ONE CROP FOR ONE DAY
# ---------------------------------------
def fetch_crop(date_str, crop_id):
    title_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
    payload = {
        "title": f"Daily Report {title_date}",
        "date": date_str,
        "commodityIds": [crop_id],
        "liveDate": "2025-11-07"
    }

    for attempt in range(5):
        try:
            r = requests.post(URL, json=payload, headers=HEADERS, timeout=60)

            if r.status_code != 200:
                log(f"  ❌ crop_id={crop_id} status={r.status_code} retry={attempt+1}")
                time.sleep(2 ** attempt)
                continue

            data = r.json()
            if "commodities" not in data:
                log(f"  ⚠️ crop_id={crop_id} returned NO data")
                return []

            commodity = data["commodities"][0]
            crop_name = commodity["items"][0]["Commodity"]
            crop_group = commodity["CommodityGroup"]

            rows = []

            for state in commodity["items"][0]["states"]:
                for market in state["markets"]:
                    for entry in market["data"]:
                        rows.append({
                            "date": date_str,
                            "Commodity": crop_name,
                            "CropGroup": crop_group,
                            "State": state["state"],
                            "Mandi": market["market_name"],

                            "Arrivals": entry["arrivals"],
                            "UnitOfArrivals": entry.get("unitOfArrivals", None),

                            "ModalPrice": entry["modalPrice"],
                            "MinPrice": entry["minPrice"],
                            "MaxPrice": entry["maxPrice"],
                            "UnitOfPrice": entry.get("unitOfPrice", None),

                            "Variety": entry["variety"],
                            "Grade": entry["grade"],
                        })

            log(f"  ✔ crop_id={crop_id} rows={len(rows)}")
            return rows

        except Exception as e:
            log(f"  ❌ crop_id={crop_id} error={e} retry={attempt+1}")
            time.sleep(2 ** attempt)

    log(f"  ❌ crop_id={crop_id} FAILED after 5 retries")
    return []


# ---------------------------------------
# FETCH ALL CROPS (parallel)
# ---------------------------------------
def fast_fetch_day(date_str):
    log(f"[{date_str}] Starting 10 crop fetches...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda cid: fetch_crop(date_str, cid), CROP_IDS)

    combined = []
    for r in results:
        combined.extend(r)

    log(f"[{date_str}] TOTAL_ROWS={len(combined)}")
    return combined


# ---------------------------------------
# SAVE TO CSV
# ---------------------------------------
def append_to_csv(rows):
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=not file_exists)


# ---------------------------------------
# MAIN LOOP (WITH RESUME CAPABILITY)
# ---------------------------------------
default_start = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 1)

current_date = load_progress(default_start)
batch = []

log("\n🚀 SCRAPER STARTED — RESUMING FROM " + current_date.strftime("%Y-%m-%d") + "\n")

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")

    day_rows = fast_fetch_day(date_str)
    batch.extend(day_rows)

    # Save progress after each day
    save_progress(date_str)

    # Flush every 40k rows
    if len(batch) >= 40_000:
        log("📦 Writing batch of 40k rows to CSV...")
        append_to_csv(batch)
        batch = []

    current_date += timedelta(days=1)

# Final flush
if batch:
    log("📦 Writing final batch...")
    append_to_csv(batch)

log("\n🎉 SCRAPING COMPLETE — FILE: ALL_CROPS_DATA.csv\n")
print("\n🎉 DONE! Check scrape.log for details.\n")
