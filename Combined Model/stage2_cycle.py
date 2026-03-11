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
    print("\n🚀 STAGE 2: CYCLE PREDICTION")
    print("=" * 80)

    if load_predictions:
        pred_path = output_dir / 'data_with_hormone_predictions.csv'
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}. Run stage1 first.")
        df = pd.read_csv(pred_path)
        print(f"Loaded data with hormone predictions from {pred_path}")
    else:
        # Alternatively, run stage1 on the fly
        print("Running stage1 to generate predictions...")
        df = run_stage1(data_dir, output_dir, force_recompute=False)

    # Load patient split
    train_patients, test_patients = data_loader.load_patient_split(output_dir)

    # Get cycle features (including predicted hormones)
    cycle_features = features.get_cycle_features(df, include_predicted=True)
    print(f"\nCycle prediction will use {len(cycle_features)} features.")

    # Prepare train/test sets
    train_cycle = df[df['id'].isin(train_patients)].dropna(subset=[config.CYCLE_TARGET])
    test_cycle = df[df['id'].isin(test_patients)].dropna(subset=[config.CYCLE_TARGET])

    print(f"Train samples: {len(train_cycle)} from {train_cycle['id'].nunique()} patients")
    print(f"Test samples: {len(test_cycle)} from {test_cycle['id'].nunique()} patients")

    X_train = train_cycle[cycle_features]
    y_train = train_cycle[config.CYCLE_TARGET]
    X_test = test_cycle[cycle_features]
    y_test = test_cycle[config.CYCLE_TARGET]

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

        # Optional: create diagnostic plots
        utils.create_regression_diagnostics(y_test, y_pred, name, config.CYCLE_TARGET,
                                            len(cycle_features), len(test_cycle['id'].unique()))

    # Save results
    res_df = pd.DataFrame(results).T
    res_df.to_csv(output_dir / 'cycle_prediction_results.csv')
    print(f"\n✅ Stage 2 completed. Results saved to {output_dir / 'cycle_prediction_results.csv'}")

    # Quick summary
    best_model = max(results, key=lambda m: results[m]['r2'])
    print(f"\n📋 Best model for cycle prediction: {best_model} (R² = {results[best_model]['r2']:.4f})")

    return results

if __name__ == "__main__":
    run_stage2()