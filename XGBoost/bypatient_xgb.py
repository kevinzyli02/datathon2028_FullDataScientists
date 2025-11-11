# =============================================================================
# COMPREHENSIVE MODEL WITH PATIENT-WISE 80/20 SPLIT
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
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time

DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")
OUTPUT_DIR = Path('patient_wise_model_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use all available files for comprehensive analysis
COMPREHENSIVE_FILES = [
    'hormones_and_selfreport.csv',
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
SAMPLE_SIZE = 15000


# =============================================================================
# EFFICIENT DATA LOADING
# =============================================================================

def load_comprehensive_data(files, data_dir, sample_size=SAMPLE_SIZE):
    """Load and merge comprehensive dataset efficiently"""
    print("📂 Loading comprehensive dataset...")

    # Start with base file
    base_file = files[0]
    df = pd.read_csv(data_dir / base_file)
    df = preprocess_data(df)

    print(f"Base data: {df.shape}")

    # Merge other files
    for file_name in files[1:]:
        print(f"🔗 Merging {file_name}...")
        try:
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"   ⚠️ File not found: {file_name}")
                continue

            new_data = pd.read_csv(file_path)
            new_data = preprocess_data(new_data)

            # Handle file structure
            new_data = handle_file_structure(new_data, file_name)

            # Find merge keys
            merge_keys = find_merge_keys(df, new_data, file_name)
            if not merge_keys:
                print(f"   ⚠️ No merge keys found for {file_name}")
                continue

            # Memory-efficient merge
            if len(new_data) > 50000:
                new_data = new_data.sample(n=20000, random_state=RANDOM_STATE)
                print(f"   🔽 Sampled new data to {new_data.shape}")

            df = pd.merge(df, new_data, on=merge_keys, how='left',
                          suffixes=('', f'_{file_name.replace(".csv", "")}'))
            print(f"✅ After {file_name}: {df.shape}")

            del new_data
            gc.collect()

        except Exception as e:
            print(f"❌ Error with {file_name}: {e}")

    # Final sampling
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


def preprocess_data(df):
    """Preprocess data with symptom mapping"""
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


# =============================================================================
# PATIENT-WISE TRAIN-TEST SPLIT
# =============================================================================

def patient_wise_train_test_split(df, test_size=0.2, random_state=42):
    """
    Split data by patient ID to prevent data leakage
    All records from a patient go to either train or test
    """
    print(f"\n🎯 Performing PATIENT-WISE 80/20 split...")

    # Get unique patient IDs
    patient_ids = df['id'].unique()
    n_patients = len(patient_ids)

    print(f"   Total patients: {n_patients}")
    print(f"   Total records: {len(df)}")

    # Split patient IDs
    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state
    )

    # Split data based on patient IDs
    train_data = df[df['id'].isin(train_patients)]
    test_data = df[df['id'].isin(test_patients)]

    print(f"   Training patients: {len(train_patients)} ({len(train_patients) / n_patients * 100:.1f}%)")
    print(f"   Testing patients:  {len(test_patients)} ({len(test_patients) / n_patients * 100:.1f}%)")
    print(f"   Training records:  {len(train_data)} ({len(train_data) / len(df) * 100:.1f}%)")
    print(f"   Testing records:   {len(test_data)} ({len(test_data) / len(df) * 100:.1f}%)")

    # Check for patient overlap
    train_ids = set(train_data['id'].unique())
    test_ids = set(test_data['id'].unique())
    overlap = train_ids.intersection(test_ids)

    if overlap:
        print(f"❌ ERROR: Patient overlap detected! {len(overlap)} patients in both sets")
        raise ValueError("Patient overlap in train/test split")
    else:
        print("✅ No patient overlap - split is valid")

    return train_data, test_data, train_patients, test_patients


# =============================================================================
# COMPREHENSIVE FEATURE PROCESSING (ALL FEATURES)
# =============================================================================

def get_all_features(df, targets):
    """Get all available features after cleaning"""
    print(f"\n🎯 Processing all available features...")

    feature_sets = {}

    for target in targets:
        if target not in df.columns:
            continue

        print(f"\nProcessing features for {target}...")

        # Remove targets and metadata
        exclude_cols = targets + ['id', 'day_in_study', 'study_interval', 'is_weekend']
        all_features = [col for col in df.columns if col not in exclude_cols]

        print(f"   Initial features: {len(all_features)}")

        # Get numeric columns only
        numeric_features = []
        for col in all_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        print(f"   Numeric features: {len(numeric_features)}")

        if len(numeric_features) == 0:
            print("❌ No numeric features found!")
            feature_sets[target] = []
            continue

        # Remove constant columns
        constant_cols = []
        for col in numeric_features:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        numeric_features = [col for col in numeric_features if col not in constant_cols]
        print(f"   Removed {len(constant_cols)} constant columns")

        # Remove high missingness columns (>80% missing)
        high_missing_cols = []
        for col in numeric_features:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.8:
                high_missing_cols.append(col)

        numeric_features = [col for col in numeric_features if col not in high_missing_cols]
        print(f"   Removed {len(high_missing_cols)} high-missing columns")

        print(f"   ✅ Final features for {target}: {len(numeric_features)}")

        # Show feature categories
        if len(numeric_features) > 0:
            categories = categorize_features(numeric_features)
            print("   Feature categories:")
            for category, count in categories.items():
                print(f"     {category}: {count}")

        feature_sets[target] = numeric_features

    return feature_sets


def categorize_features(features):
    """Categorize features by type for reporting"""
    categories = {
        'Sleep-related': len([f for f in features if 'sleep' in f.lower()]),
        'Stress-related': len([f for f in features if 'stress' in f.lower()]),
        'Heart-related': len([f for f in features if 'heart' in f.lower() or 'bpm' in f.lower()]),
        'Temperature-related': len([f for f in features if 'temp' in f.lower() or 'temperature' in f.lower()]),
        'Respiratory-related': len([f for f in features if 'respiratory' in f.lower() or 'breathing' in f.lower()]),
        'Exercise-related': len([f for f in features if 'exercise' in f.lower() or 'activity' in f.lower()]),
        'Glucose-related': len([f for f in features if 'glucose' in f.lower()]),
        'Symptom-related': len([f for f in features if any(symptom in f.lower() for symptom in [
            'appetite', 'headache', 'cramp', 'breast', 'fatigue', 'mood', 'food', 'indigestion', 'bloating'
        ])]),
        'Other': 0
    }

    categorized_count = sum(categories.values()) - categories['Other']
    categories['Other'] = len(features) - categorized_count

    return {k: v for k, v in categories.items() if v > 0}


# =============================================================================
# COMPREHENSIVE REGRESSION STATISTICS
# =============================================================================

def calculate_comprehensive_metrics(y_true, y_pred, model_name, target_name, n_features, n_patients):
    """Calculate comprehensive regression statistics"""

    print(f"\n📊 REGRESSION STATISTICS - {model_name} - {target_name}")
    print("=" * 50)
    print(f"   Features used: {n_features}")
    print(f"   Patients in set: {n_patients}")

    # Basic error metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Correlation metrics (with error handling)
    try:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    except:
        pearson_corr, pearson_p = np.nan, np.nan

    try:
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    except:
        spearman_corr, spearman_p = np.nan, np.nan

    # Additional metrics
    mean_true = np.mean(y_true)
    std_true = np.std(y_true)

    # Relative errors
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    # Print results
    print("ERROR METRICS:")
    print(f"  MAE (Mean Absolute Error):       {mae:.4f}")
    print(f"  MSE (Mean Squared Error):        {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error):  {rmse:.4f}")
    print(f"  MAPE (Mean Absolute % Error):    {mape:.2f}%")

    print("CORRELATION METRICS:")
    print(f"  R-squared (R2):                  {r2:.4f}")
    print(f"  Pearson Correlation:             {pearson_corr:.4f}")
    print(f"  Spearman Correlation:            {spearman_corr:.4f}")

    print("DATA STATISTICS:")
    print(f"  True Values - Mean: {mean_true:.4f}, Std: {std_true:.4f}")

    # Create comprehensive results dictionary
    results = {
        'model': model_name,
        'target': target_name,
        'n_features': n_features,
        'n_patients': n_patients,
        'error_metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        },
        'correlation_metrics': {
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        },
        'distribution_stats': {
            'true_mean': mean_true,
            'true_std': std_true
        }
    }

    return results


def create_regression_diagnostics(y_true, y_pred, model_name, target_name, n_features, n_patients):
    """Create comprehensive regression diagnostic plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Regression Diagnostics: {model_name} - {target_name}\nFeatures: {n_features}, Patients: {n_patients}',
        fontsize=16, fontweight='bold')

    # Plot 1: True vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('True vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(y_true, p(y_true), "b-", alpha=0.8, label=f'Regression line')
    axes[0, 0].legend()

    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Distribution comparison
    axes[1, 0].hist(y_true, bins=30, alpha=0.7, label='True Values', density=True)
    axes[1, 0].hist(y_pred, bins=30, alpha=0.7, label='Predicted Values', density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='orange')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with simple filename to avoid encoding issues
    safe_filename = f'regression_{model_name}_{target_name}'.replace(' ', '_')
    plt.savefig(OUTPUT_DIR / f'{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# COMPREHENSIVE MODEL TRAINING WITH PATIENT-WISE SPLIT
# =============================================================================

def train_comprehensive_models(df, feature_sets, targets):
    """Train models using ALL available features with PATIENT-WISE 80/20 split"""

    print("\n🚀 TRAINING COMPREHENSIVE MODELS WITH PATIENT-WISE 80/20 SPLIT")
    print("=" * 80)

    all_results = {}

    # Define models to compare
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    }

    for target in targets:
        if target not in df.columns or target not in feature_sets:
            continue

        print(f"\n🎯 TARGET: {target.upper()}")
        print("=" * 50)

        # Get ALL features for this target
        all_features = feature_sets[target]

        if len(all_features) == 0:
            print("❌ No features available for this target")
            continue

        # Prepare data
        target_data = df.dropna(subset=[target]).copy()

        print(f"📊 Dataset before split: {target_data.shape}")
        print(f"🎯 Using ALL {len(all_features)} available features")

        # PATIENT-WISE 80/20 Split
        train_data, test_data, train_patients, test_patients = patient_wise_train_test_split(
            target_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Prepare features and targets
        X_train = train_data[all_features].copy()
        y_train = train_data[target].values
        X_test = test_data[all_features].copy()
        y_test = test_data[target].values

        print(f"🔀 Final split:")
        print(f"   Training: {X_train.shape} ({len(train_patients)} patients)")
        print(f"   Testing:  {X_test.shape} ({len(test_patients)} patients)")

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        target_results = {}

        for model_name, model in models.items():
            print(f"\n🧪 Training {model_name}...")
            start_time = time.time()

            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time

                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

                # Calculate comprehensive metrics
                train_metrics = calculate_comprehensive_metrics(y_train, y_pred_train,
                                                                f"{model_name} (Train)", target,
                                                                len(all_features), len(train_patients))
                test_metrics = calculate_comprehensive_metrics(y_test, y_pred_test,
                                                               f"{model_name} (Test)", target,
                                                               len(all_features), len(test_patients))

                # Create diagnostic plots for test set
                create_regression_diagnostics(y_test, y_pred_test, model_name, target,
                                              len(all_features), len(test_patients))

                # Store results
                target_results[model_name] = {
                    'training_time': training_time,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'features_used': all_features,
                    'train_patients': train_patients,
                    'test_patients': test_patients,
                    'model': model
                }

                print(f"✅ {model_name} completed in {training_time:.2f}s")

            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                target_results[model_name] = None

        all_results[target] = target_results

        # Clean memory
        del X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        gc.collect()

    return all_results


# =============================================================================
# RESULTS COMPARISON AND REPORTING
# =============================================================================

def create_comprehensive_report(results, feature_sets):
    """Create comprehensive comparison report"""

    print("\n📋 CREATING COMPREHENSIVE RESULTS REPORT")
    print("=" * 60)

    # Save detailed results with UTF-8 encoding
    with open(OUTPUT_DIR / "patient_wise_model_report.txt", 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE MODEL ANALYSIS - PATIENT-WISE 80/20 SPLIT\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL EVALUATION WITH PATIENT-WISE TRAIN-TEST SPLIT\n")
        f.write(f"Sample Size: {SAMPLE_SIZE:,}\n")
        f.write(f"Test Size: {TEST_SIZE * 100}% of PATIENTS\n")
        f.write(f"Random State: {RANDOM_STATE}\n\n")

        for target, target_results in results.items():
            f.write(f"\nTARGET: {target.upper()}\n")
            f.write("-" * 50 + "\n")

            if target in feature_sets:
                f.write(f"ALL Features Used: {len(feature_sets[target])}\n")
                categories = categorize_features(feature_sets[target])
                f.write("Feature Categories:\n")
                for category, count in categories.items():
                    f.write(f"  {category}: {count}\n")
                f.write("\n")

            f.write("MODEL PERFORMANCE COMPARISON (TEST SET - UNSEEN PATIENTS):\n")
            f.write("-" * 60 + "\n")

            # Create comparison table
            comparison_data = []
            for model_name, result in target_results.items():
                if result and 'test_metrics' in result:
                    metrics = result['test_metrics']
                    comparison_data.append({
                        'Model': model_name,
                        'Patients': metrics['n_patients'],
                        'Features': metrics['n_features'],
                        'R2': metrics['correlation_metrics']['r2'],
                        'Pearson_r': metrics['correlation_metrics']['pearson_correlation'],
                        'Spearman_r': metrics['correlation_metrics']['spearman_correlation'],
                        'MAE': metrics['error_metrics']['mae'],
                        'RMSE': metrics['error_metrics']['rmse'],
                        'Time_s': result['training_time']
                    })

            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                f.write(comp_df.to_string(index=False, float_format='%.4f'))
                f.write("\n\n")

    # Also save results as CSV for easy analysis
    save_results_csv(results, feature_sets)

    # Create performance comparison visualization
    create_performance_comparison_plot(results)

    print(f"✅ Comprehensive report saved to {OUTPUT_DIR}")


def save_results_csv(results, feature_sets):
    """Save results as CSV files for easy analysis"""

    # Save model performance comparison
    performance_data = []
    for target, target_results in results.items():
        for model_name, result in target_results.items():
            if result and 'test_metrics' in result:
                metrics = result['test_metrics']
                performance_data.append({
                    'target': target,
                    'model': model_name,
                    'n_patients': metrics['n_patients'],
                    'n_features': metrics['n_features'],
                    'r2': metrics['correlation_metrics']['r2'],
                    'pearson_r': metrics['correlation_metrics']['pearson_correlation'],
                    'spearman_r': metrics['correlation_metrics']['spearman_correlation'],
                    'mae': metrics['error_metrics']['mae'],
                    'mse': metrics['error_metrics']['mse'],
                    'rmse': metrics['error_metrics']['rmse'],
                    'mape': metrics['error_metrics']['mape'],
                    'training_time': result['training_time']
                })

    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        perf_df.to_csv(OUTPUT_DIR / "patient_wise_performance_comparison.csv", index=False)
        print("✅ Patient-wise performance comparison saved as CSV")

    # Save feature information
    feature_data = []
    for target, features in feature_sets.items():
        categories = categorize_features(features)
        feature_data.append({
            'target': target,
            'total_features': len(features),
            **categories
        })

    if feature_data:
        feature_df = pd.DataFrame(feature_data)
        feature_df.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)
        print("✅ Feature summary saved as CSV")


def create_performance_comparison_plot(results):
    """Create performance comparison visualization"""

    for target, target_results in results.items():
        models = []
        r2_scores = []
        rmse_scores = []
        n_patients_list = []

        for model_name, result in target_results.items():
            if result and 'test_metrics' in result:
                models.append(model_name)
                r2_scores.append(result['test_metrics']['correlation_metrics']['r2'])
                rmse_scores.append(result['test_metrics']['error_metrics']['rmse'])
                n_patients_list.append(result['test_metrics']['n_patients'])

        if models:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # R² comparison
            bars1 = ax1.bar(models, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax1.set_ylabel('R² Score')
            ax1.set_title(f'Model Comparison - {target.upper()}\nTest Patients: {n_patients_list[0]}')
            ax1.set_ylim(0, 1)

            # Add value labels on bars
            for bar, score in zip(bars1, r2_scores):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.4f}', ha='center', va='bottom')

            # RMSE comparison
            bars2 = ax2.bar(models, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax2.set_ylabel('RMSE')
            ax2.set_title(f'Model Comparison - {target.upper()}\nTest Patients: {n_patients_list[0]}')

            # Add value labels on bars
            for bar, score in zip(bars2, rmse_scores):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.4f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'patient_wise_comparison_{target}.png', dpi=300, bbox_inches='tight')
            plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("🚀 COMPREHENSIVE MODEL ANALYSIS WITH PATIENT-WISE 80/20 SPLIT")
    print("=" * 80)
    print(f"📊 Models: RandomForest, XGBoost, LightGBM")
    print(f"🎯 Train-Test Split: 80%-20% OF PATIENTS (not records)")
    print(f"📈 Sample Size: {SAMPLE_SIZE:,}")
    print(f"📁 Files: {len(COMPREHENSIVE_FILES)}")
    print("=" * 80)

    try:
        # Step 1: Load comprehensive dataset
        df = load_comprehensive_data(COMPREHENSIVE_FILES, DATA_DIR, SAMPLE_SIZE)

        print(f"\n📊 Final dataset: {df.shape}")
        print(f"🎯 Targets: {TARGETS}")

        # Step 2: Get ALL available features for each target
        feature_sets = get_all_features(df, TARGETS)

        # Step 3: Train comprehensive models with PATIENT-WISE split
        results = train_comprehensive_models(df, feature_sets, TARGETS)

        # Step 4: Create comprehensive report
        create_comprehensive_report(results, feature_sets)

        print(f"\n✅ PATIENT-WISE MODEL ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"📊 Diagnostic plots and comprehensive statistics generated")

        # Print quick summary
        print(f"\n📋 QUICK SUMMARY:")
        for target, target_results in results.items():
            print(f"\n   {target.upper()}:")
            if target in feature_sets:
                print(f"     Features used: {len(feature_sets[target])}")
            best_model = None
            best_r2 = -1
            for model_name, result in target_results.items():
                if result and result['test_metrics']['correlation_metrics']['r2'] > best_r2:
                    best_r2 = result['test_metrics']['correlation_metrics']['r2']
                    best_model = model_name
            if best_model:
                test_patients = target_results[best_model]['test_metrics']['n_patients']
                print(f"     Best Model: {best_model} (R² = {best_r2:.4f})")
                print(f"     Test Patients: {test_patients}")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()