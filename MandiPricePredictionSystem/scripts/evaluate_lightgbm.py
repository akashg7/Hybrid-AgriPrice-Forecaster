from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DATA_PATH = Path("data/processed/training_data.parquet")
MODEL_PATH = Path("models/price_lgbm_model.pkl")

DATE_COL = "date"
TARGET_COL = "target_price"
TEST_FRACTION = 0.08


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def smape(y_true, y_pred):
    # Symmetric MAPE in percent
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def main():
    print(f"Loading training data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    if DATE_COL not in df.columns:
        raise ValueError(f"DATE_COL={DATE_COL} not found in columns!")

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL={TARGET_COL} not found in columns!")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n_total = len(df)
    n_test = int(n_total * TEST_FRACTION)
    if n_test < 1000:
        n_test = min(20000, n_total // 5)

    n_train = n_total - n_test
    print(f"Using last {n_test} rows as test, previous {n_train} as train/context.")

    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()

    print(f"Loading LightGBM model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    # 🔥 Use exact features from training
    model_features = list(model.feature_name_)
    print("Model was trained with features:", model_features)

    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise ValueError(f"The following model features are missing in data: {missing}")

    feature_cols = model_features
    print("Using feature columns:", feature_cols)

    X_test = df_test[feature_cols]
    y_test = df_test[TARGET_COL].to_numpy(dtype=float)

    # Naive baseline: last train target for all test
    last_train_target = float(df_train[TARGET_COL].iloc[-1])
    y_naive = np.full_like(y_test, fill_value=last_train_target, dtype=float)

    print("Predicting with LightGBM...")
    y_lgbm = model.predict(X_test)

    # --------- metrics ----------
    naive_mae = mae(y_test, y_naive)
    naive_rmse = rmse(y_test, y_naive)
    naive_mape = mape(y_test, y_naive)
    naive_smape = smape(y_test, y_naive)

    lgbm_mae = mae(y_test, y_lgbm)
    lgbm_rmse = rmse(y_test, y_lgbm)
    lgbm_mape = mape(y_test, y_lgbm)
    lgbm_smape = smape(y_test, y_lgbm)

    print("\n=== Naive baseline (global test set) ===")
    print(f"MAE   : {naive_mae:.3f}")
    print(f"RMSE  : {naive_rmse:.3f}")
    print(f"MAPE  : {naive_mape:.3f}")
    print(f"SMAPE : {naive_smape:.3f}%  (Acc ≈ {100 - naive_smape:.2f}%)")

    print("\n=== LightGBM (global test set) ===")
    print(f"MAE   : {lgbm_mae:.3f}")
    print(f"RMSE  : {lgbm_rmse:.3f}")
    print(f"MAPE  : {lgbm_mape:.3f}")
    print(f"SMAPE : {lgbm_smape:.3f}%  (Acc ≈ {100 - lgbm_smape:.2f}%)")

    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "model": ["naive", "lightgbm"],
            "mae": [naive_mae, lgbm_mae],
            "rmse": [naive_rmse, lgbm_rmse],
            "mape": [naive_mape, lgbm_mape],
            "smape": [naive_smape, lgbm_smape],
            "accuracy_pct": [100 - naive_smape, 100 - lgbm_smape],
            "n_test_rows": [len(y_test), len(y_test)],
        }
    )
    out_path = out_dir / "eval_lightgbm_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved LightGBM vs naive summary to: {out_path}")


if __name__ == "__main__":
    main()
