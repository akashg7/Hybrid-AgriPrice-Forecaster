# scripts/train_model.py
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import joblib

TRAIN_DATA_PATH = Path("data/processed/training_data.parquet")
MODEL_PATH = Path("models/price_lgbm_model.pkl")


def main():
    print(f"Loading training data from {TRAIN_DATA_PATH} ...")
    df = pd.read_parquet(TRAIN_DATA_PATH)

    # feature columns
    lag_cols = [f"lag_{l}" for l in [1, 2, 3, 7]]
    time_cols = ["dayofweek", "month"]
    cat_cols = ["Mandi", "Commodity"]

    feature_cols = lag_cols + time_cols + cat_cols

    # Train/validation split by time (no leakage)
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=60)  # last 60 days for validation

    train_mask = df["date"] <= cutoff
    valid_mask = df["date"] > cutoff

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    print(f"Train size: {train_df.shape}, Valid size: {valid_df.shape}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_price"]

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df["target_price"]

    # LightGBM can handle category dtype directly
    for col in cat_cols:
        X_train.loc[:, col] = X_train[col].astype("category")
        X_valid.loc[:, col] = X_valid[col].astype("category")

    print("Training LightGBM regressor...")
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
    )

    # Evaluate
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = sqrt(mean_squared_error(y_valid, y_pred))
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
                

if __name__ == "__main__":
    main()
