import polars as pl
import pandas as pd
import numpy as np
import gc
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

import config
import data_loader
import features
import utils

def run_stage2(data_dir=config.DATA_DIR, output_dir=config.OUTPUT_DIR, load_predictions=True):
    print("\n🚀 STAGE 2: CYCLE PREDICTION (Polars)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    if load_predictions:
        pred_path = output_dir / 'data_with_hormone_predictions.parquet'
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}. Run stage1 first.")
        df_pl = pl.read_parquet(pred_path)
        print(f"Loaded data with hormone predictions from {pred_path}")
    else:
        print("Running stage1 to generate predictions...")
        df_pl = run_stage1(data_dir, output_dir, force_recompute=False)

    # --- Check target column ---
    if config.CYCLE_TARGET not in df_pl.columns:
        print(f"\n❌ Target column '{config.CYCLE_TARGET}' not found in dataframe.")
        print("Available columns (first 20):")
        for i, col in enumerate(df_pl.columns[:20]):
            print(f"   {i+1}. {col}")
        if len(df_pl.columns) > 20:
            print(f"   ... and {len(df_pl.columns)-20} more.")
        raise KeyError(f"Column '{config.CYCLE_TARGET}' missing. Please update config.CYCLE_TARGET.")

    # Load patient split
    train_patients, test_patients = data_loader.load_patient_split(output_dir)

    # --- Print split summary ---
    total_patients = df_pl['id'].n_unique()
    train_count = len(train_patients)
    test_count = len(test_patients)
    train_records = df_pl.filter(pl.col('id').is_in(train_patients)).height
    test_records = df_pl.filter(pl.col('id').is_in(test_patients)).height
    print("\n📊 TRAIN/TEST SPLIT SUMMARY:")
    print(f"   Total patients: {total_patients}")
    print(f"   Training patients: {train_count} ({train_count/total_patients*100:.1f}%)")
    print(f"   Testing patients:  {test_count} ({test_count/total_patients*100:.1f}%)")
    print(f"   Training records:  {train_records} ({train_records/df_pl.height*100:.1f}%)")
    print(f"   Testing records:   {test_records} ({test_records/df_pl.height*100:.1f}%)")

    # For cycle prediction, we need rows with non-missing target
    train_cycle = df_pl.filter(pl.col('id').is_in(train_patients) & pl.col(config.CYCLE_TARGET).is_not_null())
    test_cycle = df_pl.filter(pl.col('id').is_in(test_patients) & pl.col(config.CYCLE_TARGET).is_not_null())

    print(f"\n🎯 Target '{config.CYCLE_TARGET}' available:")
    print(f"   Training samples with target: {train_cycle.height} (from {train_cycle['id'].n_unique()} patients)")
    print(f"   Testing samples with target:  {test_cycle.height} (from {test_cycle['id'].n_unique()} patients)")

    # Get cycle features
    cycle_features = features.get_cycle_features_polars(df_pl, include_predicted=True)
    print(f"\nCycle prediction will use {len(cycle_features)} features.")

    # Convert to pandas for modeling
    train_df = train_cycle.to_pandas()
    test_df = test_cycle.to_pandas()

    X_train = train_df[cycle_features]
    y_train = train_df[config.CYCLE_TARGET]
    X_test = test_df[cycle_features]
    y_test = test_df[config.CYCLE_TARGET]

    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_STATE, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\n🧪 Training {name} for cycle prediction...")
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'training_time': train_time
        }
        print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f} ({train_time:.2f}s)")

        utils.create_regression_diagnostics(y_test, y_pred, name, config.CYCLE_TARGET,
                                            len(cycle_features), len(test_cycle['id'].unique()))

    # Save results
    res_df = pd.DataFrame(results).T
    res_df.to_csv(output_dir / 'cycle_prediction_results.csv')
    print(f"\n✅ Stage 2 completed. Results saved to {output_dir / 'cycle_prediction_results.csv'}")

    best_model = max(results, key=lambda m: results[m]['r2'])
    print(f"\n📋 Best model for cycle prediction: {best_model} (R² = {results[best_model]['r2']:.4f})")

    return results

if __name__ == "__main__":
    run_stage2()