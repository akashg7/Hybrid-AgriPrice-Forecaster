import pandas as pd
import requests
import difflib
import logging
from io import StringIO
# from weather_collector import DistrictService # Not strictly needed if we copy overrides

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Combined Manual Overrides (Original + 22 Recovered + 7 New Reported)
# Keys are Lowercase for easier matching
MANUAL_FIXES = {
    # Original from weather_collector.py
    'banka': {'lat': 24.88, 'lon': 86.92},
    'bargarh': {'lat': 21.33, 'lon': 83.62},
    'birbhum': {'lat': 23.91, 'lon': 87.52},
    'hamirpur': {'lat': 25.95, 'lon': 80.15}, 
    'kullu': {'lat': 31.95, 'lon': 77.10},
    'mandi': {'lat': 31.58, 'lon': 76.91},
    'maharajganj': {'lat': 27.14, 'lon': 83.56},
    'paschim bardhaman': {'lat': 23.68, 'lon': 86.98},
    'pratapgarh': {'lat': 25.93, 'lon': 81.60},
    'hooghly': {'lat': 22.90, 'lon': 88.39},
    'keonjhar': {'lat': 21.63, 'lon': 85.58},
    'khandwa': {'lat': 21.83, 'lon': 76.35},
    'khargone': {'lat': 21.83, 'lon': 75.61},
    'mewat': {'lat': 28.10, 'lon': 77.00},
    'nawanshahr': {'lat': 31.12, 'lon': 76.12},
    'badaun': {'lat': 28.03, 'lon': 79.13},
    'bulandshahar': {'lat': 28.40, 'lon': 77.85},
    'coochbehar': {'lat': 26.32, 'lon': 89.45},
    'davangere': {'lat': 14.46, 'lon': 75.92},
    'delhi': {'lat': 28.61, 'lon': 77.20},
    'janjgir': {'lat': 22.01, 'lon': 82.57},
    'kanpur': {'lat': 26.44, 'lon': 80.33},
    'khurda': {'lat': 20.18, 'lon': 85.62},
    'mumbai': {'lat': 19.07, 'lon': 72.87},
    'palakad': {'lat': 10.78, 'lon': 76.65},
    'sholapur': {'lat': 17.65, 'lon': 75.90},
    'the nilgiris': {'lat': 11.41, 'lon': 76.69},
    'thiruchirappalli': {'lat': 10.79, 'lon': 78.70},
    'thirunelveli': {'lat': 8.71, 'lon': 77.75},
    'thiruvannamalai': {'lat': 12.22, 'lon': 79.07},
    'thiruvarur': {'lat': 10.76, 'lon': 79.63},
    'villupuram': {'lat': 11.94, 'lon': 79.48},
    'amarawati': {'lat': 20.93, 'lon': 77.75},
    'ambedkarnagar': {'lat': 26.41, 'lon': 82.39},
    'anupur': {'lat': 23.10, 'lon': 81.69},
    'bangalore': {'lat': 12.97, 'lon': 77.59},
    
    # The 22 Manual Recovered
    "bhadradri kothagudem": {"lat": 17.55, "lon": 80.62},
    "chattrapati sambhajinagar": {"lat": 19.88, "lon": 75.32},
    "chhota udaipur": {"lat": 22.31, "lon": 74.01},
    "cuddapah": {"lat": 14.48, "lon": 78.82},
    "deedwana kuchaman": {"lat": 27.40, "lon": 74.58},
    "dharashiv(usmanabad)": {"lat": 18.19, "lon": 76.04},
    "east jaintia hills": {"lat": 25.36, "lon": 92.37},
    "gir somnath": {"lat": 20.91, "lon": 70.37},
    "gopalgang": {"lat": 26.47, "lon": 84.43},
    "jhunjhunu": {"lat": 28.13, "lon": 75.40},
    "kotputli- behror": {"lat": 27.70, "lon": 76.20},
    "madikeri(kodagu)": {"lat": 12.43, "lon": 75.75},
    "mansa": {"lat": 29.99, "lon": 75.38},
    "neem ka thana": {"lat": 27.74, "lon": 75.78},
    "nongpoh (r-bhoi)": {"lat": 25.87, "lon": 91.83},
    "purba bardhaman": {"lat": 23.26, "lon": 87.86},
    "south west garo hills": {"lat": 25.47, "lon": 89.93},
    "south west khasi hills": {"lat": 25.32, "lon": 91.29},
    "tsemenyu": {"lat": 26.05, "lon": 94.27},
    "tuticorin": {"lat": 8.76, "lon": 78.13},
    "unokoti": {"lat": 24.33, "lon": 92.00},
    "west chambaran": {"lat": 27.15, "lon": 84.35},

    # New Fixes for Reported 0.0s
    'bhojpur': {'lat': 25.47, 'lon': 84.54},
    'bilaspur': {'lat': 22.08, 'lon': 82.15}, # CG
    'banda': {'lat': 25.49, 'lon': 80.34},
    'fatehpur': {'lat': 25.92, 'lon': 80.81},
    'lalitpur': {'lat': 24.69, 'lon': 78.42},
    'vaishali': {'lat': 25.75, 'lon': 85.42},
    'hassan': {'lat': 13.01, 'lon': 76.10}
}

def generate_all_coordinates():
    # 1. Download Master List for fuzzy matching
    url = "https://raw.githubusercontent.com/recurze/IndianCities/master/india_places.csv"
    logger.info(f"Downloading master list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        master_df = pd.read_csv(StringIO(response.text))
        master_df.columns = [c.lower() for c in master_df.columns]
        master_df['latitude'] = pd.to_numeric(master_df['latitude'], errors='coerce')
        master_df['longitude'] = pd.to_numeric(master_df['longitude'], errors='coerce')
        master_df = master_df.dropna(subset=['latitude', 'longitude'])
        
        place_map = {}
        def add_to_map(name, row):
            if not isinstance(name, str): return
            n = name.strip().lower()
            if n not in place_map:
                place_map[n] = {'lat': row['latitude'], 'lon': row['longitude']}
        for _, row in master_df.iterrows():
            if 'district' in row: add_to_map(row['district'], row)
            if 'city' in row: add_to_map(row['city'], row)
        
        possible_names = list(place_map.keys())
        logger.info(f"Loaded {len(place_map)} names for fuzzy matching.")
    except Exception as e:
        logger.error(f"Failed to load master list: {e}")
        return

    # 2. Load Targets
    districts_file = 'districts.txt'
    with open(districts_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        if lines and lines[0].lower() == 'district_name':
            targets = lines[1:]
        else:
            targets = lines
    
    logger.info(f"Loaded {len(targets)} target districts.")
    
    final_list = []
    
    for district in targets:
        d_lower = district.lower().strip()
        lat, lon = None, None
        
        # A. Manual Fixes
        if d_lower in MANUAL_FIXES:
            coords = MANUAL_FIXES[d_lower]
            lat, lon = coords['lat'], coords['lon']
            # logger.info(f"Manual override: {district}")
            
        # B. Exact Match in Place Map
        elif d_lower in place_map:
            lat = place_map[d_lower]['lat']
            lon = place_map[d_lower]['lon']
            
        # C. Fuzzy Match
        else:
            matches = difflib.get_close_matches(d_lower, possible_names, n=1, cutoff=0.6)
            if matches:
                 best = matches[0]
                 lat = place_map[best]['lat']
                 lon = place_map[best]['lon']
                 logger.info(f"Fuzzy match: {district} -> {best}")
        
        if lat is not None and lon is not None:
             if lat == 0.0 and lon == 0.0:
                 logger.warning(f"Found 0.0, 0.0 for {district}. Please verify.")
             final_list.append({'district': district, 'lat': lat, 'lon': lon})
        else:
            logger.error(f"FAILED to find coordinates for: {district}")

    # 4. Save
    out_df = pd.DataFrame(final_list)
    out_file = 'alldistrictsCoordinates.csv'
    out_df.to_csv(out_file, index=False)
    logger.info(f"Saved {len(final_list)} coordinates to {out_file}")
    
    if len(final_list) == len(targets):
        logger.info("SUCCESS: 100% Coverage.")
    else:
        logger.warning(f"WARNING: Coverage {len(final_list)}/{len(targets)}")

if __name__ == "__main__":
    generate_all_coordinates()
