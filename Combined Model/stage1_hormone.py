import pandas as pd
import numpy as np
import gc
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import config
import data_loader
import features
import utils

# =============================================================================
# OOF PREDICTION FUNCTION
# =============================================================================
def generate_oof_predictions(df, features, target, train_patients, test_patients, n_folds=5):
    """
    Generate out-of-fold predictions for training patients and predictions for test patients.
    Returns a Series with predictions for all rows (train OOF + test predictions).
    """
    print(f"   Generating OOF predictions for {target}...")
    train_idx = df['id'].isin(train_patients)
    test_idx = df['id'].isin(test_patients)

    all_preds = pd.Series(index=df.index, dtype=float)

    # Imputer and scaler will be fit on training rows with target
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_all = df[features].copy()
    train_has_target = train_idx & df[target].notna()
    X_train_has = X_all[train_has_target]
    y_train_has = df.loc[train_has_target, target]

    if len(X_train_has) == 0:
        print(f"   ⚠️ No training samples for {target}, skipping OOF.")
        return all_preds

    X_train_has_imp = imputer.fit_transform(X_train_has)
    X_train_has_scaled = scaler.fit_transform(X_train_has_imp)

    # ---- OOF for training patients ----
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    oof_preds = np.zeros(len(X_train_has))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_has_scaled)):
        X_tr, X_val = X_train_has_scaled[tr_idx], X_train_has_scaled[val_idx]
        y_tr, y_val = y_train_has.iloc[tr_idx], y_train_has.iloc[val_idx]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    all_preds.loc[train_has_target] = oof_preds

    # ---- Final model on all training data ----
    final_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_train_has_scaled, y_train_has)

    # Predict on test patients
    X_test = X_all[test_idx]
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    test_preds = final_model.predict(X_test_scaled)
    all_preds.loc[test_idx] = test_preds

    print(f"   ✅ OOF predictions completed. Train OOF size: {len(oof_preds)}, Test preds size: {len(test_preds)}")
    return all_preds

# =============================================================================
# MAIN STAGE 1 FUNCTION
# =============================================================================
def run_stage1(data_dir=config.DATA_DIR, output_dir=config.OUTPUT_DIR, force_recompute=False):
    print("\n🚀 STAGE 1: HORMONE PREDICTION WITH OOF")
    print("=" * 80)

    # 1. Load data (if filtered data exists, use it)
    filtered_dir = config.FILTERED_DATA_DIR
    if filtered_dir.exists():
        print(f"Using filtered data from {filtered_dir}")
        data_dir = filtered_dir
    else:
        # Optionally run filtering here? For simplicity, we assume data is already filtered.
        pass

    df = data_loader.load_comprehensive_data(config.COMPREHENSIVE_FILES, data_dir, config.SAMPLE_SIZE)

    # 2. Patient split (save if not exists)
    split_path = output_dir / 'patient_split.csv'
    if not split_path.exists() or force_recompute:
        train_patients, test_patients = data_loader.save_patient_split(df, config.TEST_SIZE, config.RANDOM_STATE, output_dir)
    else:
        train_patients, test_patients = data_loader.load_patient_split(output_dir)

    # 3. Baseline normalization & personalized features
    print("\n🔄 Applying baseline normalization...")
    df = features.normalize_by_baseline(df, config.TARGETS, baseline_days=3)

    print("\n🎯 Adding personalized features...")
    df = features.add_personalized_features(df, config.TARGETS, window_size=7)

    normalized_targets = [f"{t}_normalized" for t in config.TARGETS]

    # 4. Get feature sets
    feature_sets = features.get_all_features(df, normalized_targets)

    # 5. Generate OOF predictions for each hormone
    for hormone in config.TARGETS:
        norm_target = f"{hormone}_normalized"
        if norm_target not in df.columns:
            print(f"⚠️ {norm_target} not found, skipping.")
            continue
        if norm_target not in feature_sets or len(feature_sets[norm_target]) == 0:
            print(f"⚠️ No features for {norm_target}, skipping.")
            continue
        feats = feature_sets[norm_target]
        pred_col = f"{hormone}_pred"
        df[pred_col] = generate_oof_predictions(df, feats, norm_target, train_patients, test_patients)

    # 6. Save the full dataframe with predictions
    output_path = output_dir / 'data_with_hormone_predictions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Stage 1 completed. Data with predictions saved to {output_path}")

    # Optional: also run the original enhanced models for comparison? (could be a flag)
    # For now we just save predictions.

    return df

if __name__ == "__main__":
    run_stage1()