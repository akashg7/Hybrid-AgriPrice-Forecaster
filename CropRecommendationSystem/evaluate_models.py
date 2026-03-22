# CropRecommendationSystem/evaluate_models.py
# ============================================================================
# Post-Training Evaluation & Comparison Suite
# ============================================================================
# Mirrors the price forecaster's evaluate_lightgbm.py
# Loads saved models and runs comprehensive evaluation on held-out data.
# ============================================================================

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR.parent / "models"
EVAL_DIR = BASE_DIR / "outputs" / "eval"


def load_all():
    """Load models, scaler, encoders, and test data."""
    df = pd.read_parquet(DATA_DIR / "crop_features_engineered.parquet")
    encoders = joblib.load(DATA_DIR / "label_encoders.pkl")

    crop_model = joblib.load(MODEL_DIR / "crop_recommendation_lgbm.pkl")
    fert_model = joblib.load(MODEL_DIR / "fert_recommendation_lgbm.pkl")
    scaler = joblib.load(MODEL_DIR / "crop_feature_scaler.pkl")

    return df, encoders, crop_model, fert_model, scaler


def main():
    print("=" * 70)
    print("📊 CROP RECOMMENDATION: EVALUATION SUITE")
    print("=" * 70)

    df, encoders, crop_model, fert_model, scaler = load_all()

    crop_names = list(encoders['crop_type'].classes_)
    fert_names = list(encoders['fertilizer'].classes_)

    # Recreate features and test split (same seed as training)
    drop_cols = [
        'crop_type', 'fertilizer', 'soil_type',
        'crop_type_encoded', 'fertilizer_encoded',
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y_crop = df['crop_type_encoded'].values
    y_fert = df['fertilizer_encoded'].values

    from sklearn.model_selection import train_test_split
    _, X_test, _, yc_test, _, yf_test = train_test_split(
        X, y_crop, y_fert, test_size=0.20, random_state=42, stratify=y_crop
    )

    X_test_s = scaler.transform(X_test)

    # --- Crop Evaluation ---
    yc_pred = crop_model.predict(X_test_s)
    print(f"\n{'='*60}")
    print("  CROP TYPE MODEL — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy_score(yc_test, yc_pred)*100:.2f}%")
    print(f"  F1 Macro: {f1_score(yc_test, yc_pred, average='macro'):.4f}")
    print(classification_report(yc_test, yc_pred, target_names=crop_names, digits=3))

    # --- Fertilizer Evaluation ---
    yf_pred = fert_model.predict(X_test_s)
    print(f"\n{'='*60}")
    print("  FERTILIZER MODEL — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy_score(yf_test, yf_pred)*100:.2f}%")
    print(f"  F1 Macro: {f1_score(yf_test, yf_pred, average='macro'):.4f}")
    print(classification_report(yf_test, yf_pred, target_names=fert_names, digits=3))

    # --- Confusion Matrix Export ---
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    cm_crop = confusion_matrix(yc_test, yc_pred)
    cm_fert = confusion_matrix(yf_test, yf_pred)

    pd.DataFrame(cm_crop, index=crop_names, columns=crop_names).to_csv(
        EVAL_DIR / "confusion_matrix_crop.csv"
    )
    pd.DataFrame(cm_fert, index=fert_names, columns=fert_names).to_csv(
        EVAL_DIR / "confusion_matrix_fertilizer.csv"
    )
    print(f"\n💾 Confusion matrices saved to {EVAL_DIR}/")

    # --- Per-Soil-Type Breakdown ---
    print(f"\n{'='*60}")
    print("  PER-SOIL-TYPE ACCURACY BREAKDOWN")
    print(f"{'='*60}")
    soil_encoded = X_test[:, feature_cols.index('soil_type_encoded')]
    soil_le = encoders['soil_type']
    for soil_idx in sorted(np.unique(soil_encoded)):
        mask = soil_encoded == soil_idx
        if mask.sum() > 0:
            soil_name = soil_le.inverse_transform([int(soil_idx)])[0]
            crop_acc = accuracy_score(yc_test[mask], yc_pred[mask]) * 100
            fert_acc = accuracy_score(yf_test[mask], yf_pred[mask]) * 100
            print(f"  {soil_name:10s} → Crop: {crop_acc:5.1f}% | Fert: {fert_acc:5.1f}%  (n={mask.sum()})")


if __name__ == "__main__":
    main()
