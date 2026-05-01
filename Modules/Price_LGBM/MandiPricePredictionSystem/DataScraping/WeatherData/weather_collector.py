import os
import csv
import time
import json
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistrictService:
    """
    Responsible for fetching and providing the list of districts with their coordinates.
    """
    # Updated URL to a valid source
    DISTRICT_DATA_URL = "https://raw.githubusercontent.com/recurze/IndianCities/master/india_places.csv"

    # Manual coordinates for districts missing or invalid in the source
    MANUAL_OVERRIDES = {
        'banka': {'lat': 24.88, 'lon': 86.92},
        'bargarh': {'lat': 21.33, 'lon': 83.62},
        'birbhum': {'lat': 23.91, 'lon': 87.52},
        'hamirpur': {'lat': 25.95, 'lon': 80.15}, # Defaulting to UP
        'kullu': {'lat': 31.95, 'lon': 77.10},
        'mandi': {'lat': 31.58, 'lon': 76.91},
        'maharajganj': {'lat': 27.14, 'lon': 83.56},
        'paschim bardhaman': {'lat': 23.68, 'lon': 86.98},
        'pratapgarh': {'lat': 25.93, 'lon': 81.60}, # Defaulting to UP
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
    }

    # Mapping for fuzzy matches (Requested -> Available in CSV)
    NAME_MAPPING = {
        'amarawati': 'amravati',
        'ambedkarnagar': 'ambedkar nagar',
        'anupur': 'anuppur',
        'badaun': 'budaun',
        'bangalore': 'bangalore urban',
        'bulandshahar': 'bulandshahr',
        'coochbehar': 'cooch behar',
        'davangere': 'davanagere',
        'janjgir': 'janjgir-champa',
        'khurda': 'khordha',
        'mumbai': 'mumbai city',
        'palakad': 'palakkad',
        'sholapur': 'solapur',
        'the nilgiris': 'nilgiris',
        'thiruchirappalli': 'tiruchirappalli',
        'thirunelveli': 'tirunelveli',
        'thiruvannamalai': 'tiruvannamalai',
        'thiruvarur': 'tiruvarur',
        'villupuram': 'viluppuram'
    }

    def __init__(self, csv_path=None, districts_file=None):
        self.csv_path = csv_path
        self.districts_file = districts_file

    def get_districts(self):
        """
        Returns a list of dictionaries: [{'district': 'Name', 'lat': 12.3, 'lon': 76.5}, ...]
        """
        districts = []
        
        # Load target districts if provided
        target_districts = set()
        if self.districts_file and os.path.exists(self.districts_file):
            logger.info(f"Loading target districts from {self.districts_file}")
            with open(self.districts_file, 'r') as f:
                target_districts = {line.strip().lower() for line in f if line.strip()}
        
        # Load CSV Data
        if self.csv_path and os.path.exists(self.csv_path):
            logger.info(f"Loading districts from local file: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
        else:
            logger.info(f"Downloading districts from {self.DISTRICT_DATA_URL}")
            try:
                response = requests.get(self.DISTRICT_DATA_URL)
                response.raise_for_status()
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
            except Exception as e:
                logger.error(f"Failed to download district data: {e}")
                raise

        # Normalize columns
        cols = df.columns.str.lower()
        df.columns = cols
        
        # Map to standard names
        col_map = {}
        for c in df.columns:
            if 'district' in c: 
                col_map[c] = 'district'
            elif 'latitude' in c: 
                col_map[c] = 'lat'
            elif 'longitude' in c: 
                col_map[c] = 'lon'
            elif c == 'lat':
                col_map[c] = 'lat'
            elif c == 'lon' or c == 'lng':
                col_map[c] = 'lon'
            
        df = df.rename(columns=col_map)
        
        # Remove duplicate columns if any (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if not {'district', 'lat', 'lon'}.issubset(df.columns):
            # Fallback: if 'city' is present but not 'district', use city as district (approx)
            if 'city' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
                logger.warning("District column not found, using City as District.")
                df['district'] = df['city']
            else:
                logger.error(f"Missing required columns in district data. Found: {df.columns}")
                raise ValueError("District data missing required columns (district, lat, lon)")

        # Clean lat/lon: coerce to numeric, drop invalid
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        # Filter valid ranges for India
        # Approx India Bounding Box: Lat 6-38, Lon 68-98
        df = df[
            (df['lat'] >= 6) & (df['lat'] <= 38) &
            (df['lon'] >= 68) & (df['lon'] <= 98)
        ]
        
        
        # Filter out invalid district names
        df = df[df['district'] != '-']
        
        # Drop duplicates and NaNs
        # Group by district and take the first lat/lon (centroid approx)
        df = df[['district', 'lat', 'lon']].dropna().groupby('district').first().reset_index()
        
        # Create a lookup dictionary from the dataframe
        # Normalize keys to lower case for lookup
        csv_districts = {row['district'].strip().lower(): {'lat': row['lat'], 'lon': row['lon']} for _, row in df.iterrows()}
        
        # If target districts provided, iterate through THEM to preserve order and requested names
        if target_districts:
            # We iterate through the requested list (read from file again to preserve order/casing if needed, 
            # but here we have a set. Let's re-read the file to get the list)
            requested_list = []
            with open(self.districts_file, 'r') as f:
                requested_list = [line.strip() for line in f if line.strip()]
                
            for req_name in requested_list:
                req_lower = req_name.lower()
                lat, lon = None, None
                
                # 1. Check Manual Overrides
                if req_lower in self.MANUAL_OVERRIDES:
                    lat = self.MANUAL_OVERRIDES[req_lower]['lat']
                    lon = self.MANUAL_OVERRIDES[req_lower]['lon']
                
                # 2. Check CSV (Direct Match)
                elif req_lower in csv_districts:
                    lat = csv_districts[req_lower]['lat']
                    lon = csv_districts[req_lower]['lon']
                    
                # 3. Check Mapped Name in CSV
                elif req_lower in self.NAME_MAPPING:
                    mapped_name = self.NAME_MAPPING[req_lower]
                    if mapped_name in csv_districts:
                        lat = csv_districts[mapped_name]['lat']
                        lon = csv_districts[mapped_name]['lon']
                
                if lat is not None and lon is not None:
                    districts.append({
                        'district': req_name, # Use requested name
                        'lat': float(lat),
                        'lon': float(lon)
                    })
                else:
                    logger.warning(f"Could not find coordinates for requested district: {req_name}")
                    
        else:
            # If no target list, return all from CSV
            for _, row in df.iterrows():
                districts.append({
                    'district': row['district'].strip(),
                    'lat': float(row['lat']),
                    'lon': float(row['lon'])
                })
            
        logger.info(f"Loaded {len(districts)} districts.")
        return districts

class WeatherService:
    """
    Responsible for fetching weather data from NASA POWER API.
    """
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self):
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_weather(self, lat, lon, start_date, end_date):
        """
        Fetch daily weather data for a specific location and date range.
        start_date, end_date: YYYYMMDD string
        """
        # Using PRECTOTCORR (Precipitation Corrected)
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed for {lat}, {lon}: {e}")
            return None

class WeatherCollector:
    """
    Orchestrates the collection process.
    """
    def __init__(self, output_dir, start_date, end_date, districts_file=None):
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date
        self.district_service = DistrictService(districts_file=districts_file)
        self.weather_service = WeatherService()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_response(self, data, district_name):
        """
        Converts NASA POWER JSON response to a list of dicts.
        """
        if not data or 'properties' not in data or 'parameter' not in data['properties']:
            return []

        params = data['properties']['parameter']
        # NASA POWER returns data like: "T2M": {"20210101": 25.5, ...}
        
        # We need to pivot this to: date, T2M, ...
        # Get the list of dates from one of the parameters
        dates = sorted(params['T2M'].keys())
        
        records = []
        for d in dates:
            record = {
                'district': district_name,
                'date': d, # YYYYMMDD
                'temp_avg': params.get('T2M', {}).get(d),
                'temp_max': params.get('T2M_MAX', {}).get(d),
                'temp_min': params.get('T2M_MIN', {}).get(d),
                'rainfall': params.get('PRECTOTCORR', {}).get(d),
                'humidity': params.get('RH2M', {}).get(d),
                'solar_radiation': params.get('ALLSKY_SFC_SW_DWN', {}).get(d),
                'wind_speed': params.get('WS2M', {}).get(d)
            }
            records.append(record)
            
        return records

    def run(self, limit=None):
        districts = self.district_service.get_districts()
        
        if limit:
            districts = districts[:limit]
            logger.info(f"Limiting execution to first {limit} districts.")

        total = len(districts)
        logger.info(f"Starting weather collection for {total} districts from {self.start_date} to {self.end_date}")

        for i, district in enumerate(districts):
            name = district['district']
            lat = district['lat']
            lon = district['lon']
            
            # Sanitize filename
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
            filename = f"weather_{safe_name}_{self.start_date}_{self.end_date}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.exists(filepath):
                logger.info(f"[{i+1}/{total}] Skipping {name}, file exists.")
                continue
                
            logger.info(f"[{i+1}/{total}] Fetching weather for {name} ({lat}, {lon})...")
            
            data = self.weather_service.fetch_weather(lat, lon, self.start_date, self.end_date)
            
            if data:
                records = self.process_response(data, name)
                if records:
                    df = pd.DataFrame(records)
                    # Convert date to standard format YYYY-MM-DD
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {len(records)} records to {filename}")
                else:
                    logger.warning(f"No records found for {name}")
            else:
                logger.error(f"Failed to fetch data for {name}")
            
            # Rate limiting - be nice to the API
            time.sleep(1.5) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian District Weather Data Collector")
    parser.add_argument("--start-date", required=True, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYYMMDD)")
    parser.add_argument("--output-dir", default="weather_data", help="Directory to save CSV files")
    parser.add_argument("--limit", type=int, help="Limit number of districts for testing")
    parser.add_argument("--districts-file", help="Path to text file containing list of districts to process")
    
    args = parser.parse_args()
    
    collector = WeatherCollector(args.output_dir, args.start_date, args.end_date, districts_file=args.districts_file)
    collector.run(limit=args.limit)
