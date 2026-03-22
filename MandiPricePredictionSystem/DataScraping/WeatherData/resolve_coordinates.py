import pandas as pd
import requests
from io import StringIO
import difflib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resolve_coordinates():
    # 1. Download Master List
    url = "https://raw.githubusercontent.com/recurze/IndianCities/master/india_places.csv"
    logger.info(f"Downloading master list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        master_df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        logger.error(f"Failed to download master list: {e}")
        return

    # Normalize master list
    master_df.columns = [c.lower() for c in master_df.columns]
    # Ensure lat/lon are numeric
    master_df['latitude'] = pd.to_numeric(master_df['latitude'], errors='coerce')
    master_df['longitude'] = pd.to_numeric(master_df['longitude'], errors='coerce')
    master_df = master_df.dropna(subset=['latitude', 'longitude'])
    
    # Create lookup map (lowercase name -> row)
    # Priority: District -> City -> State (if name matches)
    # We'll just collect all unique names (district, city, etc)
    place_map = {}
    
    # Helper to add to map
    def add_to_map(name, row):
        if not isinstance(name, str): return
        n = name.strip().lower()
        if n not in place_map:
            place_map[n] = {'lat': row['latitude'], 'lon': row['longitude']}
            
    for _, row in master_df.iterrows():
        if 'district' in row: add_to_map(row['district'], row)
        if 'city' in row: add_to_map(row['city'], row)
        if 'state' in row: add_to_map(row['state'], row) # Less likely but possible fallback

    possible_names = list(place_map.keys())
    logger.info(f"Loaded {len(place_map)} unique place names from master list.")

    # 2. Read Missing Districts
    missing_file = 'missing_districts.txt'
    with open(missing_file, 'r') as f:
        # Skip header if present
        lines = [l.strip() for l in f if l.strip()]
        if lines and lines[0].lower() == 'district_name':
            targets = lines[1:]
        else:
            targets = lines
            
    resolved = []
    
    for district in targets:
        d_lower = district.lower()
        
        # Exact match
        if d_lower in place_map:
            match = place_map[d_lower]
            logger.info(f"Exact match: {district} -> {match}")
            resolved.append({'district': district, 'lat': match['lat'], 'lon': match['lon']})
            continue
            
        # Fuzzy match
        matches = difflib.get_close_matches(d_lower, possible_names, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            match_data = place_map[best_match]
            logger.info(f"Fuzzy match: {district} -> {best_match} ({match_data})")
            resolved.append({'district': district, 'lat': match_data['lat'], 'lon': match_data['lon']})
        else:
            logger.warning(f"No match found for: {district}")

    # 3. Save Resolved
    if resolved:
        out_df = pd.DataFrame(resolved)
        out_df.to_csv('resolved_coords.csv', index=False)
        logger.info(f"Saved {len(resolved)} resolved districts to resolved_coords.csv")
    else:
        logger.warning("No districts resolved.")

if __name__ == "__main__":
    resolve_coordinates()
