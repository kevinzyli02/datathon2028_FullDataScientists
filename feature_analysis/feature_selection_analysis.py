# =============================================================================
# MEMORY-EFFICIENT FEATURE SELECTION WITH ALL FILES AND ROBUST ERROR HANDLING
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import time

DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")
OUTPUT_DIR = Path('memory_efficient_feature_selection')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTIMIZED_FILES = ['hormones_and_selfreport.csv',  # Base file first
                   'sleep.csv',
                   'stress_score.csv',
                   'resting_heart_rate.csv',
                   'glucose.csv',
                   'computed_temperature.csv',
                   'height_and_weight.csv',
                   'exercise.csv',
                   'respiratory_rate_summary.csv',
                   'sleep_score.csv',
                   'wrist_temperature.csv'
                   ]

TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Memory optimization parameters
SAMPLE_SIZE = 20000  # Reduced for stability
CHUNK_SIZE = 5000
N_TOP_FEATURES = 15


# =============================================================================
# MEMORY-EFFICIENT DATA LOADING AND MERGING WITH ALL FILES
# =============================================================================

def load_and_merge_memory_efficient(files, data_dir, sample_size=SAMPLE_SIZE):
    """Load and merge ALL files in memory-efficient way"""
    print("📂 Loading and merging ALL files (memory-efficient)...")

    # Start with base file
    base_file = files[0]
    df = pd.read_csv(data_dir / base_file)
    df = preprocess_data(df)

    print(f"Base data: {df.shape}")

    # Process other files one by one with better memory management
    for file_name in files[1:]:
        print(f"\n🔗 Processing {file_name}...")
        try:
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"   ❌ File not found: {file_name}")
                continue

            # Load the file with optimized data types
            new_data = pd.read_csv(file_path)
            print(f"   📊 Original: {new_data.shape}")

            # Preprocess
            new_data = preprocess_data(new_data)

            # Handle different file structures
            new_data = handle_file_structure(new_data, file_name)

            # Find merge keys
            merge_keys = find_merge_keys(df, new_data, file_name)
            if not merge_keys:
                print(f"   ⚠️  No merge keys found for {file_name}")
                continue

            # Memory-efficient merge with sampling for large files
            df = memory_efficient_merge(df, new_data, merge_keys, file_name)

            # Sample if getting too large
            if len(df) > sample_size * 3:  # More aggressive sampling
                df = df.sample(n=sample_size, random_state=RANDOM_STATE)
                print(f"   🔽 Sampled to {df.shape}")

            # Clean memory
            del new_data
            gc.collect()

        except Exception as e:
            print(f"   ❌ Error processing {file_name}: {e}")
            continue

    # Final sampling to ensure manageable size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"🔽 Final sampling to: {df.shape}")

    return df


def handle_file_structure(df, file_name):
    """Handle different file structures and rename columns as needed"""

    if file_name == 'sleep.csv':
        if 'sleep_start_day_in_study' in df.columns:
            df = df.rename(columns={'sleep_start_day_in_study': 'day_in_study'})

    elif file_name == 'computed_temperature.csv':
        if 'sleep_start_day_in_study' in df.columns:
            df = df.rename(columns={'sleep_start_day_in_study': 'day_in_study'})

    elif file_name == 'exercise.csv':
        if 'start_day_in_study' in df.columns:
            df = df.rename(columns={'start_day_in_study': 'day_in_study'})

    # For high-frequency files, aggregate by participant to reduce size
    if file_name in ['glucose.csv', 'wrist_temperature.csv']:
        print(f"   📊 Aggregating {file_name} by participant...")
        df = aggregate_by_participant(df)

    return df


def aggregate_by_participant(df):
    """Aggregate high-frequency data by participant to reduce size"""
    if 'id' not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'id']

    if numeric_cols:
        # Aggregate statistics by participant
        aggregated = df.groupby('id')[numeric_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
        # Flatten column names
        aggregated.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in aggregated.columns]
        print(f"   ✅ Aggregated to {aggregated.shape}")
        return aggregated

    return df


def find_merge_keys(base_df, new_df, file_name):
    """Find appropriate merge keys between dataframes"""

    # Check for common columns
    common_cols = set(base_df.columns) & set(new_df.columns)

    if 'id' in common_cols and 'day_in_study' in common_cols:
        return ['id', 'day_in_study']
    elif 'id' in common_cols:
        return ['id']
    else:
        # Try to find alternative day columns
        day_columns = ['sleep_start_day_in_study', 'start_day_in_study', 'day_in_study']
        for day_col in day_columns:
            if day_col in new_df.columns and 'day_in_study' in base_df.columns:
                new_df = new_df.rename(columns={day_col: 'day_in_study'})
                return ['id', 'day_in_study']

        return None


def memory_efficient_merge(base_df, new_df, merge_keys, file_name):
    """Perform memory-efficient merge with sampling"""

    print(f"   🔗 Merging on: {merge_keys}")

    # If new_df is too large, sample it more aggressively
    if len(new_df) > 50000:
        sample_size = min(20000, len(new_df))
        new_df = new_df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"   🔽 Sampled new data to {new_df.shape}")

    # Perform merge
    merged = pd.merge(
        base_df,
        new_df,
        on=merge_keys,
        how='left',
        suffixes=('', f'_{file_name.replace(".csv", "")}')
    )

    print(f"   ✅ After merge: {merged.shape}")
    return merged


def preprocess_data(df):
    """Robust preprocessing that handles missing columns"""
    # Convert symptoms to numeric if they exist
    symptom_mapping = {
        'Not at all': 0, 'Very Low/Little': 1, 'Very Low': 1, 'Low': 2,
        'Moderate': 3, 'High': 4, 'Very High': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }

    symptom_cols = [
        'appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts',
        'fatigue', 'sleepissue', 'moodswing', 'stress', 'foodcravings',
        'indigestion', 'bloating'
    ]

    for col in symptom_cols:
        if col in df.columns:
            df[col] = df[col].map(symptom_mapping)

    return df


def clean_features_memory_efficient(df, targets):
    """Clean features in a memory-efficient way"""
    print("🧹 Cleaning features (memory-efficient)...")

    # Remove targets and metadata
    exclude_cols = targets + ['id', 'day_in_study', 'study_interval', 'is_weekend']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Get numeric columns only
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    print(f"   Found {len(numeric_cols)} numeric columns")

    # Remove constant columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    numeric_cols = [col for col in numeric_cols if col not in constant_cols]
    print(f"   Removed {len(constant_cols)} constant columns")

    # Remove high missingness columns
    high_missing_cols = []
    for col in numeric_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio > 0.5:
            high_missing_cols.append(col)

    numeric_cols = [col for col in numeric_cols if col not in high_missing_cols]
    print(f"   Removed {len(high_missing_cols)} high-missing columns")

    return numeric_cols


def safe_impute_memory_efficient(df, features):
    """Safe imputation that handles edge cases"""
    print("🔄 Imputing missing values...")

    # Use SimpleImputer for safe imputation
    imputer = SimpleImputer(strategy='median')

    # Ensure we only use numeric data
    X_numeric = df[features].select_dtypes(include=[np.number])

    if X_numeric.shape[1] == 0:
        print("❌ No numeric features available after cleaning")
        return pd.DataFrame()

    X_imputed = imputer.fit_transform(X_numeric)
    X_imputed = pd.DataFrame(X_imputed, columns=X_numeric.columns, index=X_numeric.index)

    print(f"   ✅ Imputation complete: {X_imputed.shape}")
    return X_imputed


# =============================================================================
# ROBUST MODEL TESTING WITH ERROR HANDLING
# =============================================================================

def test_multiple_models_safe(X_train, X_test, y_train, y_test, feature_names, target_name):
    """Test multiple models with comprehensive error handling"""

    print(f"\n🧪 Testing Models for {target_name}")
    print("=" * 50)

    results = {}

    # Define models with safe parameters
    models_config = {
        'RandomForest': {
            'model': RandomForestRegressor(
                n_estimators=50,  # Reduced for stability
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'requires_scaling': False
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'requires_scaling': False
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'requires_scaling': False
        }
    }

    successful_models = 0

    for model_name, config in models_config.items():
        print(f"   Training {model_name}...")
        start_time = time.time()

        try:
            model = config['model']

            # Check data validity
            if len(X_train) == 0 or len(y_train) == 0:
                print(f"      ❌ Empty training data for {model_name}")
                results[model_name] = None
                continue

            if np.isnan(X_train).any() or np.isnan(y_train).any():
                print(f"      ❌ NaN values in data for {model_name}")
                results[model_name] = None
                continue

            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            train_metrics = calculate_metrics_safe(y_train, y_pred_train, "Train")
            test_metrics = calculate_metrics_safe(y_test, y_pred_test, "Test")

            # Feature importance
            importance = get_feature_importance_safe(model, feature_names)

            results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'feature_importance': importance,
                'model': model
            }

            successful_models += 1
            print(f"      ✅ {model_name} - R²: {test_metrics['Test_r2']:.4f}, Time: {training_time:.2f}s")

        except Exception as e:
            print(f"      ❌ {model_name} failed: {str(e)[:100]}...")
            results[model_name] = None

    print(f"   ✅ {successful_models}/3 models trained successfully")
    return results


def calculate_metrics_safe(y_true, y_pred, prefix=""):
    """Calculate comprehensive evaluation metrics with error handling"""
    try:
        metrics = {
            f'{prefix}_r2': r2_score(y_true, y_pred),
            f'{prefix}_mse': mean_squared_error(y_true, y_pred),
            f'{prefix}_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}_mae': mean_absolute_error(y_true, y_pred)
        }

        # Calculate MAPE safely (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        metrics[f'{prefix}_mape'] = mape

    except Exception as e:
        print(f"      ⚠️  Metric calculation failed: {e}")
        metrics = {
            f'{prefix}_r2': 0.0,
            f'{prefix}_mse': 0.0,
            f'{prefix}_rmse': 0.0,
            f'{prefix}_mae': 0.0,
            f'{prefix}_mape': 0.0
        }

    return metrics


def get_feature_importance_safe(model, feature_names):
    """Safely extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return model.coef_
        else:
            return np.zeros(len(feature_names))
    except:
        return np.zeros(len(feature_names))


def create_model_comparison_plot_safe(model_results, target_name):
    """Create comparison plot with error handling"""

    try:
        # Filter out failed models
        successful_models = {}
        for model_name, result in model_results.items():
            if result is not None and 'test_metrics' in result:
                successful_models[model_name] = result

        if not successful_models:
            print(f"   ⚠️  No successful models to plot for {target_name}")
            return

        models = list(successful_models.keys())
        test_r2_scores = [successful_models[model]['test_metrics']['Test_r2'] for model in models]

        if len(models) != len(test_r2_scores):
            print(f"   ⚠️  Mismatch in model results for {target_name}")
            return

        plt.figure(figsize=(10, 6))
        colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(models)]
        bars = plt.bar(models, test_r2_scores, color=colors)

        plt.ylabel('Test R² Score')
        plt.title(f'Model Comparison for {target_name} Prediction')
        plt.ylim(0, max(1.0, max(test_r2_scores) * 1.1))

        # Add value labels on bars
        for bar, score in zip(bars, test_r2_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'model_comparison_{target_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Created model comparison plot for {target_name}")

    except Exception as e:
        print(f"   ❌ Failed to create model comparison plot: {e}")


# =============================================================================
# ROBUST FEATURE SELECTION WITH COMPREHENSIVE TESTING
# =============================================================================

def memory_efficient_feature_selection(df, targets, n_top_features=15):
    """Perform feature selection with comprehensive error handling"""

    results = {}

    for target in targets:
        print(f"\n{'=' * 60}")
        print(f"FEATURE SELECTION AND MODEL TESTING FOR: {target.upper()}")
        print(f"{'=' * 60}")

        # Filter rows where target is not missing
        target_data = df.dropna(subset=[target]).copy()

        if len(target_data) < 100:
            print(f"❌ Not enough data for {target}: only {len(target_data)} samples")
            continue

        print(f"📊 Data shape: {target_data.shape}")
        print(f"🎯 Target - Mean: {target_data[target].mean():.3f}, Std: {target_data[target].std():.3f}")

        # Clean features
        feature_cols = clean_features_memory_efficient(target_data, targets)

        if len(feature_cols) < 5:
            print(f"❌ Not enough features for {target}: only {len(feature_cols)} numeric features")
            continue

        # Impute missing values
        X_imputed = safe_impute_memory_efficient(target_data, feature_cols)

        if X_imputed.empty:
            print(f"❌ Imputation failed for {target}")
            continue

        y = target_data[target].values

        # Ensure we have enough samples after processing
        if len(X_imputed) < 50:
            print(f"❌ Not enough samples after processing for {target}: {len(X_imputed)}")
            continue

        # Split data into training (80%) and testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        print(f"🔀 Data Split:")
        print(f"   Training: {X_train.shape} ({(1 - TEST_SIZE) * 100:.0f}%)")
        print(f"   Testing:  {X_test.shape} ({TEST_SIZE * 100:.0f}%)")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # =====================================================================
        # METHOD 1: Random Forest Feature Importance
        # =====================================================================
        print(f"\n1️⃣ Random Forest Feature Importance:")

        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            rf_model.fit(X_train_scaled, y_train)

            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            top_rf_features = feature_importance.head(min(n_top_features, len(feature_importance)))['feature'].tolist()

            print(f"   ✅ Top {len(top_rf_features)} features:")
            for i, row in feature_importance.head(len(top_rf_features)).iterrows():
                print(f"   {i + 1:2d}. {row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            print(f"   ❌ Random Forest failed: {e}")
            top_rf_features = []
            feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': np.zeros(len(feature_cols))})

        # =====================================================================
        # METHOD 2: Correlation-based Selection
        # =====================================================================
        print(f"\n2️⃣ Correlation-based Feature Selection:")

        correlation_scores = []
        for feature in feature_cols:
            valid_data = target_data[[feature, target]].dropna()
            if len(valid_data) > 10:
                try:
                    corr = valid_data[feature].corr(valid_data[target])
                    if not np.isnan(corr):
                        correlation_scores.append((feature, abs(corr)))
                except:
                    continue

        if correlation_scores:
            correlation_scores.sort(key=lambda x: x[1], reverse=True)
            top_corr_features = [feat for feat, score in correlation_scores[:n_top_features]]

            print(f"   ✅ Top {len(top_corr_features)} correlated features:")
            for i, (feature, score) in enumerate(correlation_scores[:len(top_corr_features)], 1):
                print(f"   {i:2d}. {feature}: {score:.4f}")
        else:
            top_corr_features = []
            print("   ⚠️  No correlation scores calculated")

        # =====================================================================
        # METHOD 3: Combined Important Features
        # =====================================================================
        print(f"\n3️⃣ Combined Important Features:")

        combined_features = list(set(top_rf_features) | set(top_corr_features))
        if not combined_features:
            combined_features = feature_cols[:min(10, len(feature_cols))]

        print(f"   Total features selected: {len(combined_features)}")

        # =====================================================================
        # COMPREHENSIVE MODEL TESTING
        # =====================================================================
        print(f"\n4️⃣ Comprehensive Model Testing:")

        # Test with all features
        all_features_results = test_multiple_models_safe(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, target
        )

        # Create visualizations
        create_feature_importance_plot_safe(feature_importance, target)
        create_model_comparison_plot_safe(all_features_results, target)

        # Store results
        results[target] = {
            'rf_features': top_rf_features,
            'corr_features': top_corr_features,
            'combined_features': combined_features,
            'feature_importance': feature_importance,
            'all_features_results': all_features_results,
            'target_stats': {
                'mean': target_data[target].mean(),
                'std': target_data[target].std(),
                'samples': len(target_data),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }

        # Clean up memory
        del X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        gc.collect()

    return results


def create_feature_importance_plot_safe(feature_importance, target):
    """Create feature importance plot with error handling"""

    try:
        if feature_importance.empty:
            return

        plt.figure(figsize=(12, 8))

        # Plot top features
        top_features = feature_importance.head(15)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top Features for {target.upper()} Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plt.savefig(OUTPUT_DIR / f'feature_importance_{target}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Created feature importance plot for {target}")

    except Exception as e:
        print(f"   ❌ Failed to create feature importance plot: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("🚀 ROBUST FEATURE SELECTION WITH ALL FILES AND MODEL TESTING")
    print("=" * 70)
    print(f"📊 Testing with: RandomForest, XGBoost, LightGBM")
    print(f"🎯 Test Size: {TEST_SIZE * 100}%")
    print(f"📈 Sample Size: {SAMPLE_SIZE:,}")
    print(f"📁 Files to process: {len(OPTIMIZED_FILES)}")
    print("=" * 70)

    try:
        # Step 1: Load and merge ALL files efficiently
        df = load_and_merge_memory_efficient(OPTIMIZED_FILES, DATA_DIR, SAMPLE_SIZE)

        print(f"\n📊 Final dataset: {df.shape}")
        print(f"🎯 Targets: {TARGETS}")

        # Check available targets
        available_targets = [target for target in TARGETS if target in df.columns]
        print(f"✅ Available targets: {available_targets}")

        if not available_targets:
            print("❌ No target variables found!")
            return

        # Step 2: Perform feature selection with comprehensive testing
        results = memory_efficient_feature_selection(df, available_targets, N_TOP_FEATURES)

        if not results:
            print("❌ No results obtained!")
            return

        # Step 3: Save comprehensive results
        save_comprehensive_results(results)

        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")

        # Print final summary
        print(f"\n📋 FINAL SUMMARY:")
        for target, result in results.items():
            print(f"   {target.upper()}:")
            print(f"     Samples: {result['target_stats']['samples']:,}")
            print(f"     Features selected: {len(result['combined_features'])}")
            if result['all_features_results']:
                best_model = None
                best_r2 = -1
                for model_name, model_result in result['all_features_results'].items():
                    if model_result and model_result['test_metrics']['Test_r2'] > best_r2:
                        best_r2 = model_result['test_metrics']['Test_r2']
                        best_model = model_name
                print(f"     Best model: {best_model} (R²: {best_r2:.4f})")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


def save_comprehensive_results(results):
    """Save comprehensive results with model comparisons"""

    with open(OUTPUT_DIR / "comprehensive_analysis_summary.txt", 'w') as f:
        f.write("COMPREHENSIVE FEATURE SELECTION AND MODEL TESTING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        for target, result in results.items():
            f.write(f"TARGET: {target.upper()}\n")
            f.write(f"Samples: {result['target_stats']['samples']:,}\n")
            f.write(
                f"Train/Test: {result['target_stats']['train_samples']:,}/{result['target_stats']['test_samples']:,}\n")
            f.write(f"Mean: {result['target_stats']['mean']:.3f}, Std: {result['target_stats']['std']:.3f}\n\n")

            f.write("MODEL PERFORMANCE:\n")
            if result['all_features_results']:
                for model_name, model_result in result['all_features_results'].items():
                    if model_result:
                        metrics = model_result['test_metrics']
                        f.write(f"  {model_name}:\n")
                        f.write(f"    R²:  {metrics['Test_r2']:.4f}\n")
                        f.write(f"    RMSE: {metrics['Test_rmse']:.4f}\n")
                        f.write(f"    MAE:  {metrics['Test_mae']:.4f}\n")
                        f.write(f"    Time: {model_result['training_time']:.2f}s\n")
            else:
                f.write("  No model results available\n")

            f.write("\nTOP 10 FEATURES:\n")
            for i, feat in enumerate(result['combined_features'][:10], 1):
                f.write(f"  {i:2d}. {feat}\n")

            f.write("\n" + "=" * 70 + "\n\n")

    # Save combined features
    combined_features = set()
    for result in results.values():
        combined_features.update(result['combined_features'])

    pd.DataFrame({'feature': list(combined_features)}).to_csv(
        OUTPUT_DIR / "selected_features.csv", index=False)

    print(f"✅ Saved comprehensive results")
    print(f"✅ Saved {len(combined_features)} selected features")


if __name__ == "__main__":
    main()