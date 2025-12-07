import pandas as pd
import os

def check_remaining():
    districts_file = 'districts.txt'
    main_output = 'finalweather.csv'
    
    # 1. Get all target districts
    targets = set()
    with open(districts_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        if lines and lines[0].lower() == 'district_name':
            targets = set(lines[1:])
        else:
            targets = set(lines)
            
    print(f"Total target districts: {len(targets)}")

    # 2. Get present districts
    if os.path.exists(main_output):
        df = pd.read_csv(main_output)
        present = set(df['district'].unique())
        print(f"Present districts: {len(present)}")
    else:
        print("Main output file not found!")
        return

    # 3. Identify missing
    missing = sorted(list(targets - present))
    print(f"Remaining Missing Districts: {len(missing)}")
    
    if missing:
        print("List of missing districts:")
        for d in missing:
            print(f"- {d}")
    else:
        print("All districts successfully collected!")

if __name__ == "__main__":
    check_remaining()
