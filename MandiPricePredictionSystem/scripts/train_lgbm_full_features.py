# scripts/train_lgbm_full_features.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split   # ✅ added

DATA_PATH = Path("data/processed/mandi_feature_engineered.csv")  # change to .parquet if needed
MODEL_PATH = Path("models/price_lgbm_full_features.pkl")

DATE_COL = "date"
TARGET_COL = "target_price"   # we'll create this
TEST_FRACTION = 0.08


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def main():
    print(f"Loading engineered data from {DATA_PATH} ...")

    # 0) Load data
    if DATA_PATH.suffix == ".csv":
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_parquet(DATA_PATH)

    print("Columns:", df.columns.tolist())
    print("Shape before target:", df.shape)

    # 1) Parse date & sort by (series_id, date)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(["series_id", DATE_COL]).reset_index(drop=True)

    # 2) Create target: next-day ModalPrice per series
    df[TARGET_COL] = (
        df.groupby("series_id")["ModalPrice"]
          .shift(-1)
    )

    # Drop rows where we don't know the future price
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print("Shape after adding target & dropping NaNs:", df.shape)

    # 3) Time-based split (global)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n_total = len(df)
    n_test = int(n_total * TEST_FRACTION)
    if n_test < 1000:
        n_test = min(20000, n_total // 5)

    n_train = n_total - n_test
    print(f"Using last {n_test} rows as test, previous {n_train} as train.")

    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()

    # 4) Choose feature columns (all engineered features minus obvious non-features)
    drop_cols = [
        DATE_COL,
        TARGET_COL,
        "ModalPrice",        # current price (we're predicting next)
        "Commodity",
        "CropGroup",
        "State",
        "Mandi",
        "Variety",
        "Grade",
        "district_name",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    print("Number of feature columns:", len(feature_cols))
    print("Example feature cols:", feature_cols[:15])

    X_train = df_train[feature_cols].copy()
    y_train = df_train[TARGET_COL].to_numpy(dtype=float)

    X_test = df_test[feature_cols].copy()
    y_test = df_test[TARGET_COL].to_numpy(dtype=float)

    # 5) Mark categorical columns (id columns) as category
    cat_cols = [c for c in feature_cols if c.endswith("_id")]
    print("Categorical feature columns:", cat_cols)

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # 6) Train/val split inside train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=False
    )

    print("Training LightGBM with validation split...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        max_depth=-1,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[]
    )

    # 7) Evaluate vs naive baseline on test set
    print("Predicting on test set...")
    y_lgbm = model.predict(X_test)

    # Naive baseline: global last train target for comparison
    last_train_target = float(y_train[-1])
    y_naive = np.full_like(y_test, fill_value=last_train_target, dtype=float)

    # Metrics
    naive_mae = mae(y_test, y_naive)
    naive_rmse = rmse(y_test, y_naive)
    naive_mape = mape(y_test, y_naive)
    naive_smape = smape(y_test, y_naive)

    lgbm_mae = mae(y_test, y_lgbm)
    lgbm_rmse = rmse(y_test, y_lgbm)
    lgbm_mape = mape(y_test, y_lgbm)
    lgbm_smape = smape(y_test, y_lgbm)

    print("\n=== Naive baseline (full-feature dataset) ===")
    print(f"MAE   : {naive_mae:.3f}")
    print(f"RMSE  : {naive_rmse:.3f}")
    print(f"MAPE  : {naive_mape:.3f}")
    print(f"SMAPE : {naive_smape:.3f}%  (Acc ≈ {100 - naive_smape:.2f}%)")

    print("\n=== LightGBM FULL FEATURES ===")
    print(f"MAE   : {lgbm_mae:.3f}")
    print(f"RMSE  : {lgbm_rmse:.3f}")
    print(f"MAPE  : {lgbm_mape:.3f}")
    print(f"SMAPE : {lgbm_smape:.3f}%  (Acc ≈ {100 - lgbm_smape:.2f}%)")

    # 8) Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved full-feature LightGBM model to: {MODEL_PATH}")

    # 9) Save per-row test predictions + summary
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-row predictions
    test_out = df_test.copy()
    test_out["y_true"] = y_test
    test_out["y_pred_lgbm"] = y_lgbm
    test_out["y_pred_naive"] = y_naive

    preds_path = out_dir / "lgbm_full_features_test_predictions.csv"
    test_out.to_csv(preds_path, index=False)
    print(f"Saved per-row test predictions to: {preds_path}")

    # summary metrics
    summary = pd.DataFrame(
        {
            "model": ["naive_full", "lightgbm_full"],
            "mae": [naive_mae, lgbm_mae],
            "rmse": [naive_rmse, lgbm_rmse],
            "mape": [naive_mape, lgbm_mape],
            "smape": [naive_smape, lgbm_smape],
            "accuracy_pct": [100 - naive_smape, 100 - lgbm_smape],
            "n_test_rows": [len(y_test), len(y_test)],
        }
    )
    out_path = out_dir / "eval_lightgbm_full_features_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved full-feature LightGBM eval summary to: {out_path}")


if __name__ == "__main__":
    main()
