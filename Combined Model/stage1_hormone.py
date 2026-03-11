import pandas as pd
import polars as pl
import numpy as np
import gc
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb
import config
import data_loader
import features
import utils

def generate_oof_predictions_pandas(df_pd, features, target, train_patients, test_patients, n_folds=5):
    """Original OOF function that expects pandas DataFrame."""
    print(f"   Generating OOF predictions for {target}...")
    train_idx = df_pd['id'].isin(train_patients)
    test_idx = df_pd['id'].isin(test_patients)

    all_preds = pd.Series(index=df_pd.index, dtype=float)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_all = df_pd[features].copy()
    train_has_target = train_idx & df_pd[target].notna()
    X_train_has = X_all[train_has_target]
    y_train_has = df_pd.loc[train_has_target, target]

    if len(X_train_has) == 0:
        print(f"   ⚠️ No training samples for {target}, skipping OOF.")
        return all_preds

    X_train_has_imp = imputer.fit_transform(X_train_has)
    X_train_has_scaled = scaler.fit_transform(X_train_has_imp)

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

    final_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_train_has_scaled, y_train_has)

    X_test = X_all[test_idx]
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    test_preds = final_model.predict(X_test_scaled)
    all_preds.loc[test_idx] = test_preds

    print(f"   ✅ OOF predictions completed.")
    return all_preds

def run_stage1(data_dir=config.DATA_DIR, output_dir=config.OUTPUT_DIR, force_recompute=False):
    print("\n🚀 STAGE 1: HORMONE PREDICTION WITH OOF (Polars)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data with Polars
    df_pl = data_loader.load_data_polars(config.COMPREHENSIVE_FILES, data_dir, config.SAMPLE_SIZE, use_parquet=True)

    # 2. Patient split (save using pandas version – convert to pandas)
    df_pd_for_split = df_pl.select(['id']).unique().to_pandas()
    # Actually, we need the full df for split, but save_patient_split only needs IDs. We can pass the Polars df and it converts.
    train_patients, test_patients = data_loader.save_patient_split(df_pl, config.TEST_SIZE, config.RANDOM_STATE, output_dir)

    # 3. Baseline normalization & personalized features (Polars)
    print("\n🔄 Applying baseline normalization...")
    df_pl = features.normalize_by_baseline_polars(df_pl, config.TARGETS, baseline_days=3)

    print("\n🎯 Adding personalized features...")
    df_pl = features.add_personalized_features_polars(df_pl, config.TARGETS, window_size=7)

    normalized_targets = [f"{t}_normalized" for t in config.TARGETS]

    # 4. Get feature sets (Polars)
    feature_sets = features.get_all_features_polars(df_pl, normalized_targets)

    # 5. Convert to pandas for OOF generation
    df_pd = df_pl.to_pandas()

    # 6. Generate OOF predictions for each hormone
    for hormone in config.TARGETS:
        norm_target = f"{hormone}_normalized"
        if norm_target not in df_pd.columns:
            continue
        if norm_target not in feature_sets or len(feature_sets[norm_target]) == 0:
            continue
        feats = feature_sets[norm_target]
        pred_col = f"{hormone}_pred"
        df_pd[pred_col] = generate_oof_predictions_pandas(df_pd, feats, norm_target, train_patients, test_patients)

    # 7. Convert back to Polars and save
    df_pl_final = pl.from_pandas(df_pd)
    output_path = output_dir / 'data_with_hormone_predictions.parquet'
    df_pl_final.write_parquet(output_path)
    print(f"\n✅ Stage 1 completed. Data with predictions saved to {output_path} (Parquet)")

    # Also save as CSV for compatibility? Optional.
    df_pl_final.write_csv(output_dir / 'data_with_hormone_predictions.csv')
    return df_pl_final

if __name__ == "__main__":
    run_stage1()