import os
import pandas as pd
import glob

def merge_weather_data():
    source_dir = 'weather_data'
    output_file = 'WeatherData.csv'
    
    # Get all CSV files in the source directory
    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Found {len(csv_files)} files to merge.")
    
    # List to store dataframes
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure district column exists, if not derive from filename?
            # Based on inspection, 'district' column already exists in the files.
            # But just in case, we can check.
            if 'district' not in df.columns:
                # Filename format: weather_DistrictName_dates.csv
                basename = os.path.basename(file)
                parts = basename.split('_')
                if len(parts) >= 2:
                    district_name = parts[1] # Assumes weather_DistrictName_...
                    df['district'] = district_name
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # Move 'district' to the first column if it's not
        cols = ['district'] + [c for c in merged_df.columns if c != 'district']
        merged_df = merged_df[cols]
        
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged {len(all_data)} files into {output_file}")
        print(f"Total rows: {len(merged_df)}")
    else:
        print("No data merged.")

if __name__ == "__main__":
    # Change to the directory where the script is located to ensure paths are relative to it
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    merge_weather_data()
