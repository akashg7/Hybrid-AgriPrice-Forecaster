import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

###############################################################
# 1. STATE ID → STATE NAME MAPPING (OFFICIAL)
###############################################################
state_names = {
    1: "ANDAMAN AND NICOBAR ISLANDS",
    2: "ANDHRA PRADESH",
    3: "ARUNACHAL PRADESH",
    4: "ASSAM",
    5: "BIHAR",
    6: "CHANDIGARH",
    7: "CHHATTISGARH",
    8: "DADRA AND NAGAR HAVELI",
    9: "DAMAN AND DIU",
    10: "DELHI",
    11: "GOA",
    12: "GUJARAT",
    13: "HARYANA",
    14: "HIMACHAL PRADESH",
    15: "JAMMU AND KASHMIR",
    16: "JHARKHAND",
    17: "KARNATAKA",
    18: "KERALA",
    19: "LADAKH",
    20: "LAKSHADWEEP",
    21: "MADHYA PRADESH",
    22: "MAHARASHTRA",
    23: "MANIPUR",
    24: "MEGHALAYA",
    25: "MIZORAM",
    26: "NAGALAND",
    27: "ODISHA",
    28: "PUDUCHERRY",
    29: "PUNJAB",
    30: "RAJASTHAN",
    31: "SIKKIM",
    32: "TAMIL NADU",
    33: "TELANGANA",
    34: "TRIPURA",
    35: "UTTAR PRADESH",
    36: "UTTARAKHAND",
    # MISSING IN YOUR API, BUT INCLUDED HERE:
    37: "WEST BENGAL"
}


###############################################################
# 2. FIXED: FETCH DISTRICTS
###############################################################
def fetch_districts(state_id):
    url = f"https://api.agmarknet.gov.in/v1/guest-location-filters?state_id={state_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        districts = r.json()   # LIST
        for d in districts:
            d["state_id"] = state_id
            d["state_name"] = state_names.get(state_id, "UNKNOWN")   # ADD STATE NAME
        return districts
    except Exception as e:
        print(f"State {state_id} failed → {e}")
        return []


###############################################################
# 3. FIXED: FETCH MARKETS FOR A DISTRICT
###############################################################
def fetch_markets(state_id, district):
    district_id = district["id"]
    district_name = district["district_name"]
    state_name = state_names.get(state_id, "UNKNOWN")

    url = f"https://api.agmarknet.gov.in/v1/guest-location-filters?state_id={state_id}&district_id={district_id}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        markets = r.json()  # LIST
    except:
        markets = []

    results = []
    for m in markets:
        results.append({
            "state_id": state_id,
            "state_name": state_name,         # ADD STATE NAME
            "district_id": district_id,
            "district_name": district_name,
            "market_id": m.get("id"),
            "market_name": m.get("mkt_name"),
        })
    
    return results


###############################################################
# 4. FETCH ALL DISTRICTS
###############################################################
all_districts = []

with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {
        ex.submit(fetch_districts, sid): sid
        for sid in range(1, 37)
    }
    
    for fut in as_completed(futures):
        sid = futures[fut]
        try:
            districts = fut.result()
            print(f"State {sid}: {len(districts)} districts")
            all_districts.extend(districts)
        except Exception as e:
            print(f"State {sid} failed: {e}")


###############################################################
# 5. FETCH ALL MARKETS FOR ALL DISTRICTS
###############################################################
all_markets = []

with ThreadPoolExecutor(max_workers=30) as ex:
    futures = {}

    for d in all_districts:
        sid = d["state_id"]
        futures[ex.submit(fetch_markets, sid, d)] = (sid, d["id"])
    
    for fut in as_completed(futures):
        sid, did = futures[fut]
        try:
            rows = fut.result()
            print(f"State {sid}, District {did}: {len(rows)} markets")
            all_markets.extend(rows)
        except Exception as e:
            print(f"District {did} failed: {e}")


###############################################################
# 6. SAVE FINAL OUTPUT WITH STATE NAME INCLUDED
###############################################################
mandi_master = pd.DataFrame(all_markets)
mandi_master.to_csv("agmarknet_master_parallel.csv", index=False)

print("\nSaved → agmarknet_master_parallel.csv")
print(mandi_master.head())
print("\nShape:", mandi_master.shape)
