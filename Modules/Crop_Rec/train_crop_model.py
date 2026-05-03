# CropRecommendationSystem/train_crop_model.py
# ============================================================================
# Crop Recommendation Training Pipeline
# ============================================================================
# Mirrors the price forecaster's train_lgbm_full_features.py:
#   - Proper stratified train/test splits
#   - Multiple model comparison (LightGBM vs RandomForest)
#   - Full evaluation suite (accuracy, macro-F1, per-class metrics)
#   - Naive baseline comparison
#   - 5-fold stratified cross-validation
#   - Feature importance analysis
#   - Model + artifact export
# ============================================================================

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
FEATURE_DATA_PATH = DATA_DIR / "crop_features_engineered.parquet"
MODEL_DIR = BASE_DIR.parent / "models"
EVAL_DIR = BASE_DIR / "outputs" / "eval"


# ============================================================================
# Metrics (same style as price forecaster)
# ============================================================================
def evaluate_model(name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  F1 (macro)   : {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    return {
        'model': name,
        'accuracy': round(acc * 100, 2),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
    }


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    print("=" * 70)
    print("🌾 CROP RECOMMENDATION: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Engineered Data
    if not FEATURE_DATA_PATH.exists():
        print(f"❌ Engineered data not found at {FEATURE_DATA_PATH}")
        print("   Run 'python build_features.py' first!")
        return

    df = pd.read_parquet(FEATURE_DATA_PATH)
    print(f"✅ Loaded engineered data: {df.shape}")

    # 2. Define feature columns (drop raw text, targets, encoded targets)
    drop_cols = [
        'crop_type',
        'crop_type_encoded',
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"   Using {len(feature_cols)} feature columns")

    X = df[feature_cols].values
    y_crop = df['crop_type_encoded'].values

    # Load encoders for class names
    encoders = joblib.load(DATA_DIR / "label_encoders.pkl")
    crop_names = list(encoders['crop_type'].classes_)

    # 3. Stratified Train/Test Split (80/20)
    X_train, X_test, yc_train, yc_test = train_test_split(
        X, y_crop, test_size=0.20, random_state=42, stratify=y_crop
    )
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ========================================================================
    # 5. CROP TYPE — Multi-Model Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("📌 TARGET 1: CROP TYPE PREDICTION")
    print("=" * 70)

    # --- Naive Baseline (always predict most common class) ---
    from collections import Counter
    most_common_crop = Counter(yc_train).most_common(1)[0][0]
    y_naive_crop = np.full_like(yc_test, fill_value=most_common_crop)
    naive_crop_metrics = evaluate_model("Naive Baseline (Most Frequent)", yc_test, y_naive_crop, crop_names)

    # --- Random Forest ---
    print("\n⚙️  Training Random Forest for Crop...")
    rf_crop = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_crop.fit(X_train_s, yc_train)
    yc_pred_rf = rf_crop.predict(X_test_s)
    rf_crop_metrics = evaluate_model("Random Forest (Crop)", yc_test, yc_pred_rf, crop_names)

    # --- LightGBM ---
    print("\n⚙️  Training LightGBM for Crop...")
    lgbm_crop = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_crop.fit(X_train_s, yc_train)
    yc_pred_lgbm = lgbm_crop.predict(X_test_s)
    lgbm_crop_metrics = evaluate_model("LightGBM (Crop)", yc_test, yc_pred_lgbm, crop_names)

    # --- Cross-Validation for LightGBM Crop ---
    print("🔄 5-Fold Stratified Cross-Validation (LightGBM Crop)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lgbm_crop, X_train_s, yc_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    lgbm_crop_metrics['cv_accuracy_mean'] = round(cv_scores.mean() * 100, 2)
    lgbm_crop_metrics['cv_accuracy_std'] = round(cv_scores.std() * 100, 2)

    # Pick best crop model
    best_crop_model = lgbm_crop if lgbm_crop_metrics['accuracy'] >= rf_crop_metrics['accuracy'] else rf_crop
    best_crop_name = "LightGBM" if best_crop_model is lgbm_crop else "RandomForest"
    print(f"\n🏆 Best Crop Model: {best_crop_name}")

    # ========================================================================
    # 6. Feature Importance Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 FEATURE IMPORTANCE (Top 15 for Crop Prediction)")
    print("=" * 70)
    importances = lgbm_crop.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for i, row in feat_imp.head(15).iterrows():
        bar = '█' * int(row['importance'] / feat_imp['importance'].max() * 30)
        print(f"   {row['feature']:30s} {row['importance']:6.0f}  {bar}")

    # ========================================================================
    # 7. Save All Artifacts
    # ========================================================================
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Models
    joblib.dump(best_crop_model, MODEL_DIR / "crop_recommendation_lgbm.pkl")
    joblib.dump(scaler, MODEL_DIR / "crop_feature_scaler.pkl")

    print(f"\n💾 Saved models to {MODEL_DIR}/")

    # Evaluation summary CSV
    summary = pd.DataFrame([
        naive_crop_metrics, rf_crop_metrics, lgbm_crop_metrics,
    ])
    summary_path = EVAL_DIR / "model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"💾 Saved evaluation summary to {summary_path}")

    # Feature importance CSV
    feat_imp_path = EVAL_DIR / "feature_importance_crop.csv"
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"💾 Saved feature importance to {feat_imp_path}")

    # Per-row test predictions
    test_preds = pd.DataFrame({
        'y_true_crop': yc_test,
        'y_pred_crop_lgbm': yc_pred_lgbm,
        'y_pred_crop_rf': yc_pred_rf,
    })
    preds_path = EVAL_DIR / "test_predictions.csv"
    test_preds.to_csv(preds_path, index=False)
    print(f"💾 Saved per-row test predictions to {preds_path}")

    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
