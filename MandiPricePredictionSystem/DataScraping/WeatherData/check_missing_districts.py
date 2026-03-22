import os
from weather_collector import DistrictService

def check_missing():
    # 1. Load requested districts
    with open('target_districts.txt', 'r') as f:
        requested = {line.strip().lower() for line in f if line.strip()}
    
    # 2. Load available districts using the service
    # We pass None for districts_file so it loads ALL available districts from the CSV source
    service = DistrictService()
    available_districts_list = service.get_districts()
    available = {d['district'].lower() for d in available_districts_list}
    
    # 3. Find missing
    missing = requested - available
    
    print(f"Requested: {len(requested)}")
    print(f"Available in source: {len(available)}")
    print(f"Matched: {len(requested - missing)}")
    print(f"Missing: {len(missing)}")
    print("\n--- Missing Districts ---")
    for m in sorted(missing):
        print(m)

if __name__ == "__main__":
    check_missing()
