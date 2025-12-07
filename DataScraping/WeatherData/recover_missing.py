import pandas as pd
import os
import argparse
from parallel_weather_collector import ParallelWeatherCollector

def recover_missing_data():
    main_output = 'finalweather.csv'
    recovery_output = 'recovery_weather.csv'
    resolved_coords = 'resolved_coords.csv'
    
    if not os.path.exists(resolved_coords):
        print(f"{resolved_coords} not found. Run resolve_coordinates.py first.")
        return

    print(f"Starting recovery run using {resolved_coords}...")
    
    # We pass the CSV path both as the districts file (to satisfy the arg) AND as csv_path
    # The Collector logic will look for 'district', 'lat', 'lon' in the CSV if provided as districts_file and csv_path?
    # Actually, DistrictService logic: if csv_path is provided, it loads it.
    # If districts_file is provided, it filters by it.
    # But here resolved_coords.csv HAS the target districts.
    # So we can just use it as BOTH.
    
    # However, ParallelWeatherCollector init expects districts_file to be a list of names.
    # If we pass resolved_coords.csv as districts_file, it might try to read lines as names.
    # Let's create a temporary list of names from resolved_coords.csv to be safe.
    
    df_resolved = pd.read_csv(resolved_coords)
    temp_list = 'resolved_districts_list.txt'
    with open(temp_list, 'w') as f:
        f.write("district_name\n")
        for d in df_resolved['district']:
            f.write(f"{d}\n")
            
    collector = ParallelWeatherCollector(
        output_file=recovery_output,
        start_date='20240101',
        end_date='20251201',
        districts_file=temp_list,
        max_workers=10, 
        csv_path=resolved_coords
    )
    collector.run()
    
    # Merge
    if os.path.exists(recovery_output):
        df_rec = pd.read_csv(recovery_output)
        print(f"Recovered {len(df_rec)} rows.")
        if not df_rec.empty:
            df_main = pd.read_csv(main_output)
            merged = pd.concat([df_main, df_rec], ignore_index=True)
            # Deduplicate just in case
            merged = merged.drop_duplicates(subset=['district', 'date'])
            merged.to_csv(main_output, index=False)
            print(f"Merged successfully. Total rows: {len(merged)}")
            # Cleanup
            if os.path.exists(recovery_output): os.remove(recovery_output)
            if os.path.exists(temp_list): os.remove(temp_list)
        else:
            print("Recovery run yielded no data.")
    else:
        print("Recovery run produced no output file.")

if __name__ == "__main__":
    recover_missing_data()
