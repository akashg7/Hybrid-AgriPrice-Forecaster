import difflib
from weather_collector import DistrictService

def find_matches():
    # 1. Load requested districts
    with open('target_districts.txt', 'r') as f:
        requested = {line.strip().lower() for line in f if line.strip()}
    
    # 2. Load available districts
    service = DistrictService()
    available_districts_list = service.get_districts()
    available = {d['district'].lower() for d in available_districts_list}
    
    # 3. Find missing
    missing = requested - available
    
    print(f"Missing: {len(missing)}")
    print("\n--- Suggested Matches ---")
    for m in sorted(missing):
        # Find closest match
        matches = difflib.get_close_matches(m, available, n=1, cutoff=0.6)
        if matches:
            print(f"'{m}': '{matches[0]}',")
        else:
            print(f"'{m}': None,")

if __name__ == "__main__":
    find_matches()
