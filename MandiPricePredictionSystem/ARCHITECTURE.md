# Hybrid-AgriPrice-Forecaster Architecture Flow

Based on a detailed scan of the repository, here is the comprehensive structural overview and architectural flow of the Hybrid-AgriPrice-Forecaster project.

## 1. System Architecture Diagram

```mermaid
flowchart TD
    %% Define Data Sources
    subgraph Data_Collection ["Data Collection (Raw Sources)"]
        A1["Agmarknet<br>(Daily Mandi Prices)"] -->|"Scraped Data"| B1("DataScraping Scripts")
        A2["Geographic API/Database<br>(District Lat/Lon Mappings)"] --> B2("Mandi-Coordinate Map")
        A3["NASA POWER API<br>(Historical Weather Data)"] -->|"Temp, Rainfall, etc."| B3("Weather Enrichment")
    end

    %% Data Linking
    B1 --> C1{"MergingFiles<br>& PreProcessing"}
    B2 --> B3
    B3 --> C1

    %% Post Collection Raw Data
    C1 -->|"merged_crop_data_with_weather.csv<br>mandi_crop_pairs.csv"| D1("Feature Engineering Phase")

    subgraph Feature_Engineering ["Data Processing & Feature Engineering"]
        D1 --> D2["build_training_data.py"]
        D1 --> D3["prepare_moirai_data.py"]
        
        D2 -->|"Parquet Data"| E1(("Processed Data<br>Lags, Rolling Stats,<br>Weather Anomalies"))
        D3 -->|"Scaled/Formatted Data"| E2(("Moirai Scaled Data"))
    end

    subgraph Model_Training ["Model Training"]
        E1 --> F1("LightGBM Full Features<br>'train_lgbm_full_features.py'")
        E2 --> F2("Salesforce Moirai<br>Zero-Shot Base Model")
        
        F1 -->|"Models & Weights"| G1[("LightGBM PKL")]
        F2 -->|"Uses 'uni2ts' Library"| G2[("Pre-trained Moirai")]
    end

    subgraph Forecasting_Engine ["Batch & Single Forecasting"]
        G1 -.->|"forecast_all.py<br>forecast_single.py"| H1("LightGBM Forecasts")
        G2 -.->|"forecast_all_moirai.py<br>moirai_forecast_single.py"| H2("Moirai Forecasts")
        E1 -.-> H1
        E2 -.-> H2
        H1 --> H3(("Forecast Outputs<br>14-Day Price Predictions"))
        H2 --> H3
    end

    subgraph Analytics_Layer ["Analytics & Actionable Insights"]
        H3 --> I1["recommend_market.py<br>(Best Market Recommender)"]
        H3 --> I2["analytics_nearby_markets.py<br>(Nearby Price Comparison)"]
        H3 --> I3["best_crop_by_region.py<br>(Crop Recommendation)"]
        H3 --> I4["query_price_forecast.py<br>(Direct Queries)"]
    end

    %% Styling
    classDef source fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef datastore fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef outputs fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;

    class A1,A2,A3 source;
    class B1,B2,B3,C1,D2,D3 process;
    class E1,E2,G1,G2 datastore;
    class I1,I2,I3,I4,H3 outputs;
```

---

## 2. Component Breakdown

### 2.1. Ingestion / Data Collection Pipeline
- `DataScraping/`: Contains the logic used to ping and systematically scrape **Agmarknet** day-by-day.
- **Geospatial Mapping**: Raw Mandi districts are fused with Latitude/Longitude coordinates so they can communicate seamlessly with weather APIs.
- **Weather Enrichment**: Utilizing the exact coordinates, the pipeline pulls deeply granular metrics (temp bounds, solar radiation, humidity, rainfall) from the **NASA POWER API**.
- `MergingFiles/` & `DataPreProcessing/`: Cleans overlapping datasets and outputs the foundation: `merged_crop_data_with_weather.csv`.

### 2.2. Feature Engineering & Preparation Stack (`scripts/`)
- `build_training_data.py`: Handles massive dimensionality expansion for ML (expanding ~20 variables to 70+). Calculates time/seasonal embeddings, 1-30 day lag prices, localized market volatility, and rolling statistical means.
- `prepare_moirai_data.py`: Tailors and scales inputs specifically to comply with the architectural requirements of the unified time-series (`uni2ts`) inputs for Salesforce Moirai.

### 2.3. Model Training Ecosystem
- **Primary Architecture (LightGBM)**: Uses `train_lgbm_full_features.py` to compile the decision tree ensemble model (`price_lgbm_full_features.pkl`). Currently achieving the strongest predictive metrics against the test set constraints (~90.3% SMAPE Inv).
- **Secondary/Experimental (Salesforce Moirai)**: Integrates with the `uni2ts` subdirectory to execute a state-of-the-art transformer foundation model acting entirely on zero-shot inference (no direct downstream fine-tuning).

### 2.4. Forecasting Block
- Utilizes `forecast_all.py` and `forecast_all_moirai.py` to batch traverse through `mandi_crop_pairs.csv` mapping arrays. Outputs high-fidelity, 14-day future horizon models stored directly into the `outputs/forecasts/` directory logic.

### 2.5. Consumer & Analytics Layer
Abstracts the mathematical forecasts into real-world insights:
- **`recommend_market.py`**: Tells a farmer exactly where they should route their supply for max profit locally.
- **`analytics_nearby_markets.py`**: Helps contrast current spikes & historical baselines against neighboring geographic centers.
- **`best_crop_by_region.py`**: Evaluates which of the "Top 10" designated crops yield the safest/highest price trajectory organically depending on the spatial coordinates.

---

## 3. Top 10 Target Strategy Framework
The entire pipeline isn't just a brute-force approach on every single Indian commodity format. By architecture design, it actively partitions against specific behaviors:
1. **High Volatility Items**: Onion, Tomato (Excellent anomalous spike testing).
2. **Standard MSP Staples**: Wheat, Rice (Robust baseline cycle testing).
3. **Oil/Secondary**: Mustard, Groundnut, Potato, Chili, Banana, Watermelon.
