# Dataset Information & Sources

This document details the data sources, collection methods, and processing pipelines used to construct the final dataset for the Agricultural Mandi Price Forecasting project.

## 1. Primary Data Sources

It contains all crop data and mandi mapping and coordinates and feature engineered data -
https://drive.google.com/drive/folders/1WS1Nhvwz97tAhNzfUtnW7DKwKsFnbpBp?usp=sharing

Merged Crop Data with Weather Data -
https://drive.google.com/file/d/150IkSQip5hQelhY-Upy-g741uLlvx4MZ/view?usp=sharing

### A. Agricultural Prices & Arrivals (Agmarknet)
- **Source**: [Agmarknet (Government of India)](https://agmarknet.gov.in/)
- **API Endpoint**: `https://api.agmarknet.gov.in/v1/prices-and-arrivals/commodity-market/daily-report-weighted`
- **Method**: Custom Python scraper (`DataScraping/CropData/fast_scrape.py`)
- **Coverage**:
  - **Period**: Jan 1, 2024 – Dec 1, 2025
  - **Granularity**: Daily
  - **Scope**: Top 10 Commodities across all Mandis in India.

### B. Weather Data (NASA POWER)
- **Source**: [NASA POWER Project](https://power.larc.nasa.gov/)
- **API Endpoint**: `https://power.larc.nasa.gov/api/temporal/daily/point`
- **Method**: Parallelized collector (`DataScraping/WeatherData/weather_collector.py`)
- **Parameters Fetched**:
  - `T2M`: Temperature at 2 Meters
  - `T2M_MAX`: Maximum Temperature at 2 Meters
  - `T2M_MIN`: Minimum Temperature at 2 Meters
  - `PRECTOTCORR`: Precipitation (Corrected)
  - `RH2M`: Relative Humidity at 2 Meters
  - `ALLSKY_SFC_SW_DWN`: All Sky Surface Shortwave Downward Irradiance (Solar Radiation)
  - `WS2M`: Wind Speed at 2 Meters

### C. Geographic Data (District Coordinates)
- **Source**: [IndianCities Database (GitHub)](https://raw.githubusercontent.com/recurze/IndianCities/master/india_places.csv)
- **Method**: Fuzzy matching + Manual overrides (`DataScraping/WeatherData/generate_all_coordinates.py`)
- **Purpose**: To map each district to a specific Latitude/Longitude for querying weather APIs.

## 2. Data Collection Process

### Step 1: Crop Data Scraping
We scraped daily market data for 10 specific crops. Since Agmarknet does not provide a bulk download, we reverse-engineered their internal API to fetch data day-by-day.
- **Script**: `DataScraping/CropData/fast_scrape.py`
- **Logic**: Iterates through each date and crop ID, handling retries and saving to `ALL_CROPS_DATA.csv`.

### Step 2: District Mapping
Agmarknet data provides "District Name" but no coordinates. Since the weather API requires Lat/Lon, we created a mapping layer.
- **Script**: `DataScraping/WeatherData/generate_all_coordinates.py`
- **Process**:
  1. Extracted unique district names from crop data.
  2. Matched against the `IndianCities` database.
  3. Applied manual fixes for 30+ mismatched names (e.g., "Banaras" -> "Varanasi", "Prayagraj" -> "Allahabad").
  4. **Output**: `alldistrictsCoordinates.csv`

### Step 3: Weather Data Enrichment
For every unique district in the dataset, we fetched historical daily weather.
- **Script**: `DataScraping/WeatherData/weather_collector.py`
- **Process**:
  - For each district, queried NASA API for the full 2-year range.
  - Merged this weather data back into the main crop dataset based on `Date` and `District`.

## 3. Feature Engineering

The raw data contained ~20 columns. We expanded this to **70+ features** to capture complex market dynamics.

### Temporal Features
- `day`, `month`, `year`, `day_of_week`, `week_of_year`, `day_of_year`
- `is_weekend` (Binary flag)
- **Cyclical Encodings**: `sin1`, `cos1`, `sin2`, `cos2` (Seasonal fourier terms)

### Price Dynamics
- **Lags**: `modal_lag_1`, `modal_lag_3`, `modal_lag_7`, `modal_lag_14`, `modal_lag_30`
- **Rolling Stats**: Mean & Std Dev for 3, 7, 14, 30 days window.
- **Volatility**: `volatility_7`, `volatility_30`, `zscore_7`
- **Structure**: `price_range` (Max - Min), `price_spread`, `momentum_7`

### Supply Dynamics (Arrivals)
- **Lags**: `arrivals_lag_1` ... `arrivals_lag_14`
- **Averages**: `arrivals_avg_7`, `arrivals_avg_30`
- **Shock**: `arrival_change_7`

### Environmental Impact
- `temp_anomaly`: Deviation from 30-day average.
- `rain_anomaly`: Deviation from 30-day average.
- `rain_intensity`, `temp_range`

### Spatial Embeddings
- `lat_sin`, `lat_cos`, `lon_sin`, `lon_cos` (To help models learn regional proximity)

## 4. Key Files & Links List

| Description | Link / Path |
|-------------|-------------|
| **Raw Crop Data Scraper** | [fast_scrape.py](file:///DataScraping/CropData/fast_scrape.py) |
| **Agmarknet API** | `https://api.agmarknet.gov.in/v1/prices-and-arrivals/commodity-market/daily-report-weighted` |
| **Weather Collector** | [weather_collector.py](file:///DataScraping/WeatherData/weather_collector.py) |
| **NASA POWER API** | `https://power.larc.nasa.gov/api/temporal/daily/point` |
| **Coordinate Generator** | [generate_all_coordinates.py](file:///DataScraping/WeatherData/generate_all_coordinates.py) |
| **Indian Cities Database** | `https://raw.githubusercontent.com/recurze/IndianCities/master/india_places.csv` |
