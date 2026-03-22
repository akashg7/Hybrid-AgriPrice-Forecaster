import os
import argparse
import pandas as pd
import concurrent.futures
from weather_collector import WeatherCollector, WeatherService, DistrictService
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelWeatherCollector:
    def __init__(self, output_file, start_date, end_date, districts_file, max_workers=20, csv_path=None):
        self.output_file = output_file
        self.start_date = start_date
        self.end_date = end_date
        self.districts_file = districts_file
        self.max_workers = max_workers
        self.district_service = DistrictService(districts_file=districts_file, csv_path=csv_path)
        self.weather_service = WeatherService()

    def fetch_district_weather(self, district):
        name = district['district']
        lat = district['lat']
        lon = district['lon']
        
        try:
            # logger.info(f"Fetching weather for {name}...")
            data = self.weather_service.fetch_weather(lat, lon, self.start_date, self.end_date)
            if data:
                # Reuse the process_response logic from WeatherCollector but we need to instantiate it or simple copy logic
                # Since WeatherCollector logic is simple, let's just use a helper here or instantiate on the fly?
                # Using a dummy instance to reuse method is also fine, or just static method key.
                # Let's just create a temporary collector instance to reuse logic if possible, 
                # OR better, just duplicate the simple parsing logic to avoid overhead/dependency issues.
                
                if 'properties' not in data or 'parameter' not in data['properties']:
                    return []

                params = data['properties']['parameter']
                dates = sorted(params['T2M'].keys())
                
                records = []
                for d in dates:
                    record = {
                        'district': name,
                        'date': pd.to_datetime(d, format='%Y%m%d').strftime('%Y-%m-%d'),
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
            else:
                logger.warning(f"No data returned for {name}")
                return []
        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")
            return []

    def run(self):
        districts = self.district_service.get_districts()
        logger.info(f"Found {len(districts)} districts to process.")
        
        all_records = []
        completed = 0
        total = len(districts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_district = {executor.submit(self.fetch_district_weather, d): d for d in districts}
            
            for future in concurrent.futures.as_completed(future_to_district):
                district = future_to_district[future]
                try:
                    records = future.result()
                    if records:
                        all_records.extend(records)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total} districts processed. collected {len(all_records)} rows.")
                except Exception as exc:
                    logger.error(f"{district['district']} generated an exception: {exc}")

        if all_records:
            logger.info(f"Saving {len(all_records)} records to {self.output_file}...")
            df = pd.DataFrame(all_records)
            # Ensure column order
            cols = ['district', 'date', 'temp_avg', 'temp_max', 'temp_min', 'rainfall', 'humidity', 'solar_radiation', 'wind_speed']
            df = df[cols]
            df.to_csv(self.output_file, index=False)
            logger.info("Done.")
        else:
            logger.warning("No records collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Weather Collector")
    parser.add_argument("--start-date", default="20230606", help="YYYYMMDD")
    parser.add_argument("--end-date", default="20250606", help="YYYYMMDD")
    parser.add_argument("--output-file", default="finalwetherdata.csv", help="Output CSV file")
    parser.add_argument("--districts-file", default="districts.txt", help="Input districts file")
    parser.add_argument("--workers", type=int, default=20, help="Number of threads")
    
    parser.add_argument("--csv-path", help="Path to local district CSV file for coordinates")
    
    args = parser.parse_args()
    
    collector = ParallelWeatherCollector(args.output_file, args.start_date, args.end_date, args.districts_file, args.workers, args.csv_path)
    collector.run()
