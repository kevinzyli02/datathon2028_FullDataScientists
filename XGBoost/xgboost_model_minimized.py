# sequential_variable_analysis.py
"""
SEQUENTIAL VARIABLE ANALYSIS
Process files one at a time to find best variables from entire dataset
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('sequential_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of files to process sequentially
ALL_FILES = [
    'hormones_and_selfreport.csv',  # Base file first
    'active_minutes.csv',
    'sleep.csv',
    'stress_score.csv',
    #'steps.csv',
    'resting_heart_rate.csv',
    'glucose.csv',
    'heart_rate_variability_details.csv',
    'computed_temperature.csv',
    #'demographic_vo2_max.csv',
    'height_and_weight.csv',
    #'subject-info.csv'
]
#include active zone, calories, estimated oxygen variation, exercise, respiratory rate, sleep score, wrist temperature
#remove steps, demographic vo2 max, subject info, calories
TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# SEQUENTIAL PROCESSING FUNCTIONS
# =============================================================================

def load_base_dataset():
    """Load the base hormones dataset"""
    print("📂 Loading base dataset...")
    base_file = 'hormones_and_selfreport.csv'
    file_path = DATA_DIR / base_file

    if not file_path.exists():
        raise ValueError(f"Base file {base_file} not found")

    base_data = pd.read_csv(file_path)
    print(f"   ✅ Base data: {base_data.shape}")
    return base_data


def load_and_merge_single_file(base_data, file_name):
    """Load and merge a single file with the current base data"""
    print(f"\n🔄 Processing {file_name}...")

    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"   ❌ {file_name}: Not found")
        return base_data

    try:
        new_data = pd.read_csv(file_path)
        print(f"   ✅ {file_name}: {new_data.shape}")

        # Define merge strategy for this file
        if file_name == 'hormones_and_selfreport.csv':
            return base_data  # Already loaded as base

        elif file_name == 'sleep.csv':
            # Sleep data uses sleep_start_day_in_study
            if 'sleep_start_day_in_study' in new_data.columns:
                new_data = new_data.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
            merged_data = pd.merge(base_data, new_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name in ['demographic_vo2_max.csv', 'height_and_weight.csv', 'subject-info.csv']:
            # Demographic data - merge on id only
            merged_data = pd.merge(base_data, new_data, on=['id'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name == 'heart_rate_variability_details.csv':
            # Aggregate HRV data first
            aggregated_data = aggregate_hrv_data_simple(new_data)
            merged_data = pd.merge(base_data, aggregated_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        else:
            # Default merge on id and day_in_study
            merged_data = pd.merge(base_data, new_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        print(f"   ✅ After {file_name}: {merged_data.shape}")
        return merged_data

    except Exception as e:
        print(f"   ❌ Error processing {file_name}: {e}")
        return base_data


def aggregate_hrv_data_simple(df):
    """Simple aggregation for HRV data"""
    if 'day_in_study' not in df.columns:
        return df

    hrv_metrics = ['rmssd', 'low_frequency', 'high_frequency']
    available_metrics = [col for col in hrv_metrics if col in df.columns]

    if not available_metrics:
        return df

    # Simple mean aggregation
    aggregated = df.groupby(['id', 'day_in_study'])[available_metrics].mean().reset_index()
    aggregated = aggregated.rename(columns={col: f'{col}_mean' for col in available_metrics})

    return aggregated


def preprocess_data(df):
    """Fast preprocessing for the current dataset"""
    # Convert symptoms to numeric
    symptom_mapping = {'Not at all': 0, 'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
    symptom_cols = [col for col in ['stress', 'fatigue', 'appetite', 'exerciselevel'] if col in df.columns]

    for col in symptom_cols:
        df[col] = df[col].map(symptom_mapping)

    # Handle categorical variables quickly
    categorical_cols = [col for col in ['phase', 'gender'] if col in df.columns]
    for col in categorical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
            df[col] = df[col].replace(-1, np.nan)

    return df


def evaluate_current_variables(df, current_best_variables=None):
    """Evaluate variables in current dataset and return best ones"""
    print("   🎯 Evaluating variables...")

    # Get all potential predictors
    exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study']
    all_predictors = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Filter for variables with reasonable completeness
    valid_predictors = []
    for col in all_predictors:
        completeness = df[col].notna().sum() / len(df)
        if completeness > 0.1:  # At least 10% complete
            valid_predictors.append(col)

    print(f"   📊 Testing {len(valid_predictors)} variables")

    if len(valid_predictors) < 3:
        print("   ⚠️  Not enough variables to evaluate")
        return current_best_variables or []

    # Prepare data
    X = df[valid_predictors].copy()
    y = df[TARGETS].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) < 50:
        print("   ⚠️  Not enough samples for evaluation")
        return current_best_variables or []

    # Simple correlation-based ranking (faster than XGBoost)
    variable_scores = []

    for variable in valid_predictors:
        scores = []
        for target_idx, target in enumerate(TARGETS):
            # Calculate correlation on complete cases
            valid_data = pd.concat([X[variable], y.iloc[:, target_idx]], axis=1).dropna()
            if len(valid_data) > 10:
                corr = valid_data[variable].corr(valid_data.iloc[:, 1])
                if not np.isnan(corr):
                    scores.append(abs(corr))

        if scores:
            avg_score = np.mean(scores)
            variable_scores.append((variable, avg_score))

    # Sort by average correlation
    variable_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top 20 variables
    top_variables = [var for var, score in variable_scores[:20]]

    print(f"   ✅ Selected top {len(top_variables)} variables")

    # Combine with previous best variables
    if current_best_variables:
        combined_variables = list(set(current_best_variables + top_variables))
        print(f"   📈 Total variables so far: {len(combined_variables)}")
        return combined_variables
    else:
        return top_variables


def final_evaluation_with_test_set(df, best_variables):
    """Final evaluation with train/test split using accumulated best variables"""
    print(f"\n🏆 FINAL EVALUATION with {len(best_variables)} best variables")

    if not best_variables:
        print("   ⚠️  No variables to evaluate")
        return {}

    # Prepare data
    X = df[best_variables].copy()
    y = df[TARGETS].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"   📊 Final dataset: {X.shape}")

    if len(X) < 100:
        print("   ⚠️  Not enough data for final evaluation")
        return {}

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train final model
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    model.fit(X_train_imputed, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_imputed)
    y_test_pred = model.predict(X_test_imputed)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Individual target performance
    target_results = {}
    for i, target in enumerate(TARGETS):
        target_results[target] = {
            'train_r2': r2_score(y_train.iloc[:, i], y_train_pred[:, i]),
            'test_r2': r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        }

    # Feature importance
    feature_importance = calculate_feature_importance(model, best_variables)

    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'target_results': target_results,
        'feature_importance': feature_importance,
        'n_features': len(best_variables),
        'n_samples': len(X)
    }

    print(f"   ✅ Final Model Performance:")
    print(f"      Overall - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    for target in TARGETS:
        print(
            f"      {target} - Train R²: {target_results[target]['train_r2']:.3f}, Test R²: {target_results[target]['test_r2']:.3f}")

    return results


def calculate_feature_importance(model, feature_names):
    """Calculate feature importance from trained model"""
    importance_data = []

    for i, estimator in enumerate(model.estimators_):
        importance = estimator.feature_importances_
        for j, (feature, imp) in enumerate(zip(feature_names, importance)):
            importance_data.append({
                'hormone': TARGETS[i],
                'variable': feature,
                'importance': imp
            })

    importance_df = pd.DataFrame(importance_data)

    # Calculate average importance across all hormones
    avg_importance = importance_df.groupby('variable')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)
    avg_importance['importance_pct'] = avg_importance['importance'] / avg_importance['importance'].sum() * 100

    return avg_importance


def create_final_report(results, best_variables, processing_history):
    """Create final report and visualizations"""
    print("\n📋 Creating final report...")

    if not results:
        print("   ⚠️  No results to report")
        return

    # Plot 1: Feature importance
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    top_features = results['feature_importance'].head(15)
    plt.barh(range(len(top_features)), top_features['importance_pct'])
    plt.yticks(range(len(top_features)), top_features['variable'])
    plt.xlabel('Feature Importance (%)')
    plt.title('Top 15 Most Important Variables')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    # Plot 2: Model performance
    plt.subplot(2, 2, 2)
    metrics = ['Train R²', 'Test R²']
    scores = [results['train_r2'], results['test_r2']]
    bars = plt.bar(metrics, scores, color=['lightblue', 'lightgreen'])
    plt.ylabel('R² Score')
    plt.title('Overall Model Performance')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    # Plot 3: Target-specific performance
    plt.subplot(2, 2, 3)
    targets = list(results['target_results'].keys())
    train_scores = [results['target_results'][t]['train_r2'] for t in targets]
    test_scores = [results['target_results'][t]['test_r2'] for t in targets]

    x_pos = np.arange(len(targets))
    width = 0.35

    plt.bar(x_pos - width / 2, train_scores, width, label='Train R²', alpha=0.7)
    plt.bar(x_pos + width / 2, test_scores, width, label='Test R²', alpha=0.7)
    plt.xlabel('Target Variable')
    plt.ylabel('R² Score')
    plt.title('Performance by Target Variable')
    plt.xticks(x_pos, targets)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Processing history
    plt.subplot(2, 2, 4)
    files = [entry['file'] for entry in processing_history]
    n_variables = [entry['n_variables'] for entry in processing_history]

    plt.plot(range(len(files)), n_variables, marker='o', linewidth=2)
    plt.xlabel('Processing Step')
    plt.ylabel('Number of Variables')
    plt.title('Variable Accumulation During Processing')
    plt.xticks(range(len(files)), [f.split('.')[0][:10] for f in files], rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sequential_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    results['feature_importance'].to_csv(OUTPUT_DIR / 'final_feature_importance.csv', index=False)

    # Save variable list
    variable_df = pd.DataFrame({'variable': best_variables})
    variable_df.to_csv(OUTPUT_DIR / 'best_variables_list.csv', index=False)

    # Save processing history
    history_df = pd.DataFrame(processing_history)
    history_df.to_csv(OUTPUT_DIR / 'processing_history.csv', index=False)

    # Create summary report
    report = f"""
SEQUENTIAL VARIABLE ANALYSIS REPORT
====================================

PROCESSING SUMMARY:
- Files processed: {len(processing_history)}
- Final variables selected: {len(best_variables)}
- Total samples in final model: {results['n_samples']}

FINAL MODEL PERFORMANCE:
- Overall Train R²: {results['train_r2']:.3f}
- Overall Test R²:  {results['test_r2']:.3f}

TARGET-SPECIFIC PERFORMANCE:
{chr(10).join([f'- {target}: Train R² = {results['target_results'][target]['train_r2']:.3f}, Test R² = {results['target_results'][target]['test_r2']:.3f}' for target in TARGETS])}

TOP 10 VARIABLES:
{chr(10).join([f'{i + 1:2d}. {row["variable"]:<30} : {row["importance_pct"]:5.2f}%' for i, row in results['feature_importance'].head(10).iterrows()])}

PROCESSING HISTORY:
{chr(10).join([f'- {entry["file"]}: {entry["n_variables"]} variables' for entry in processing_history])}
"""

    with open(OUTPUT_DIR / 'sequential_analysis_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return report


# =============================================================================
# MAIN SEQUENTIAL PROCESSING
# =============================================================================

def main():
    """Main sequential processing function"""
    print("=" * 80)
    print("SEQUENTIAL VARIABLE ANALYSIS - PROCESSING FILES ONE BY ONE")
    print("=" * 80)

    processing_history = []
    best_variables = []
    current_data = None

    try:
        # Step 1: Load base dataset
        current_data = load_base_dataset()
        current_data = preprocess_data(current_data)

        # Step 2: Process each file sequentially
        for file_name in ALL_FILES:
            # Skip base file (already loaded)
            if file_name == 'hormones_and_selfreport.csv':
                # Evaluate base variables
                best_variables = evaluate_current_variables(current_data)
                processing_history.append({
                    'file': file_name,
                    'n_variables': len(best_variables),
                    'data_shape': current_data.shape
                })
                continue

            # Load and merge next file
            current_data = load_and_merge_single_file(current_data, file_name)
            current_data = preprocess_data(current_data)

            # Evaluate current variables
            best_variables = evaluate_current_variables(current_data, best_variables)

            # Record processing step
            processing_history.append({
                'file': file_name,
                'n_variables': len(best_variables),
                'data_shape': current_data.shape
            })

            # Clean memory
            gc.collect()

        print(f"\n✅ Sequential processing completed!")
        print(f"   Total variables accumulated: {len(best_variables)}")
        print(f"   Final dataset shape: {current_data.shape}")

        # Step 3: Final evaluation with test set
        final_results = final_evaluation_with_test_set(current_data, best_variables)

        # Step 4: Create final report
        if final_results:
            create_final_report(final_results, best_variables, processing_history)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📁 Results saved to: {OUTPUT_DIR}/")
        print("🎯 Use best_variables_list.csv for your final model features")

        return final_results, best_variables

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, variables = main()