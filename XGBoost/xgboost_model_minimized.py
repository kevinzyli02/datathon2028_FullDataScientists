# robust_sequential_analysis.py
"""
ROBUST SEQUENTIAL ANALYSIS
Handles files with different column structures and missing merge keys
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('robust_sequential_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of files to process sequentially
ALL_FILES = ['hormones_and_selfreport.csv',  # Base file first

             # Original key files
             # 'active_minutes.csv',
             'sleep.csv',
             'stress_score.csv',
             # 'steps.csv',
             'resting_heart_rate.csv',
             'glucose.csv',
             #'heart_rate_variability_details.csv',
             'computed_temperature.csv',
             # 'demographic_vo2_max.csv',
             'height_and_weight.csv',
             # 'subject-info.csv',

             # NEWLY REQUESTED FILES
             # 'active_zone_minutes.csv',
             # 'calories.csv',
             # 'estimated_oxygen_variation.csv',
             'exercise.csv',
             'respiratory_rate_summary.csv',
                'sleep_score.csv',
             'wrist_temperature.csv'
             ]
TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# ROBUST FILE PROCESSING FUNCTIONS
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


def inspect_file_columns(file_name):
    """Inspect a file's columns to understand its structure"""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"   ❌ {file_name}: File not found")
        return None

    try:
        # Read just the first row to get columns
        sample = pd.read_csv(file_path, nrows=5)
        print(f"   📋 {file_name} columns: {list(sample.columns)}")
        print(f"   📊 {file_name} shape: {sample.shape}")
        return sample
    except Exception as e:
        print(f"   ❌ Error inspecting {file_name}: {e}")
        return None


def find_merge_columns(file_data, file_name):
    """Find appropriate columns to use for merging"""
    print(f"   🔍 Finding merge columns for {file_name}...")

    # Priority list of column names to look for
    possible_id_columns = ['id', 'subject_id', 'participant_id', 'user_id']
    possible_day_columns = ['day_in_study', 'sleep_start_day_in_study', 'start_day_in_study', 'day', 'date']

    id_column = None
    day_column = None

    # Find ID column
    for col in possible_id_columns:
        if col in file_data.columns:
            id_column = col
            break

    # Find day column
    for col in possible_day_columns:
        if col in file_data.columns:
            day_column = col
            break

    # If no day column found but we have an ID, we can still merge on ID only
    if id_column:
        print(f"   ✅ Found ID column: {id_column}")
        if day_column:
            print(f"   ✅ Found day column: {day_column}")
            return [id_column, day_column]
        else:
            print(f"   ⚠️  No day column found, will merge on {id_column} only")
            return [id_column]
    else:
        print(f"   ❌ No ID column found in {file_name}")
        return None


def load_and_merge_single_file_robust(base_data, file_name):
    """Robust file loading and merging with automatic column detection"""
    print(f"\n🔄 Processing {file_name}...")

    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"   ❌ {file_name}: File not found")
        return base_data

    try:
        # First inspect the file structure
        file_sample = inspect_file_columns(file_name)
        if file_sample is None:
            return base_data

        # Load the full file
        new_data = pd.read_csv(file_path)
        print(f"   ✅ {file_name}: {new_data.shape}")

        # Special handling for high-frequency files
        if file_name in ['wrist_temperature.csv', 'estimated_oxygen_variation.csv']:
            new_data = process_high_frequency_file_safe(file_path, file_name)
            if new_data is None:
                return base_data

        # Find merge columns for this file
        merge_columns = find_merge_columns(new_data, file_name)

        if merge_columns is None:
            print(f"   ⚠️  Cannot merge {file_name} - no suitable columns found")
            return base_data

        # Handle special cases
        if file_name == 'hormones_and_selfreport.csv':
            return base_data  # Already loaded

        elif file_name in ['sleep.csv', 'respiratory_rate_summary.csv', 'sleep_score.csv', 'computed_temperature.csv']:
            # For sleep-related files, they might use sleep_start_day_in_study
            if 'sleep_start_day_in_study' in new_data.columns and 'day_in_study' not in new_data.columns:
                new_data = new_data.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
                # Update merge columns if needed
                if 'sleep_start_day_in_study' in merge_columns:
                    merge_columns = ['id', 'day_in_study']

        elif file_name == 'exercise.csv':
            # Exercise data uses start_day_in_study
            if 'start_day_in_study' in new_data.columns and 'day_in_study' not in new_data.columns:
                new_data = new_data.rename(columns={'start_day_in_study': 'day_in_study'})
                if 'start_day_in_study' in merge_columns:
                    merge_columns = ['id', 'day_in_study']

        # Check if all merge columns exist in both datasets
        missing_in_base = [col for col in merge_columns if col not in base_data.columns]
        missing_in_new = [col for col in merge_columns if col not in new_data.columns]

        if missing_in_base:
            print(f"   ⚠️  Merge columns {missing_in_base} missing in base data")
            return base_data

        if missing_in_new:
            print(f"   ⚠️  Merge columns {missing_in_new} missing in {file_name}")
            return base_data

        # Perform the merge
        print(f"   🔗 Merging on columns: {merge_columns}")
        merged_data = pd.merge(
            base_data,
            new_data,
            on=merge_columns,
            how='left',
            suffixes=('', f'_{file_name.replace(".csv", "")}')
        )

        print(f"   ✅ After {file_name}: {merged_data.shape}")
        return merged_data

    except Exception as e:
        print(f"   ❌ Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return base_data


def process_high_frequency_file_safe(file_path, file_name):
    """Safe processing for high-frequency files with proper column detection"""
    print(f"   🔍 Processing high-frequency file: {file_name}")

    try:
        # First inspect the file structure
        sample = pd.read_csv(file_path, nrows=1000)
        print(f"   📋 Sample columns: {list(sample.columns)}")

        # Identify value and timestamp columns
        value_column = None
        timestamp_column = None

        # Look for value columns (temperature, oxygen, etc.)
        value_keywords = ['value', 'temp', 'temperature', 'oxygen', 'hrv', 'hr']
        for col in sample.columns:
            if any(keyword in col.lower() for keyword in value_keywords) and col != 'id':
                value_column = col
                break

        # Look for timestamp columns
        time_keywords = ['timestamp', 'datetime', 'time', 'date']
        for col in sample.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                timestamp_column = col
                break

        # If no specific value column found, use first numeric column that's not ID
        if not value_column:
            numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'id']
            if numeric_cols:
                value_column = numeric_cols[0]

        print(f"   🎯 Using value column: {value_column}")
        print(f"   🕒 Using timestamp column: {timestamp_column}")

        if not value_column or not timestamp_column:
            print(f"   ⚠️  Could not identify required columns, reading file normally")
            return pd.read_csv(file_path)

        # Process in chunks for memory efficiency
        chunk_size = 50000
        all_aggregated = []

        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            print(f"      Processing chunk {i + 1}...")

            # Clean and aggregate chunk
            chunk_clean = clean_data_chunk(chunk, value_column, timestamp_column)
            if chunk_clean is not None and not chunk_clean.empty:
                aggregated = aggregate_to_daily(chunk_clean, value_column, timestamp_column)
                if aggregated is not None and not aggregated.empty:
                    all_aggregated.append(aggregated)

            # Limit chunks for very large files
            if i >= 3:  # Process only 200K rows
                print(f"      ⚠️  Limiting to first 200K rows")
                break

        if all_aggregated:
            final_aggregated = pd.concat(all_aggregated, ignore_index=True)

            # Final aggregation to handle any duplicates
            final_daily = final_aggregated.groupby('id').agg('mean').reset_index()

            print(f"   ✅ Aggregated {file_name} to {final_daily.shape}")
            return final_daily
        else:
            print(f"   ⚠️  No data aggregated, returning sample")
            return sample

    except Exception as e:
        print(f"   ❌ Error in high-frequency processing: {e}")
        # Fallback: read the file normally
        try:
            return pd.read_csv(file_path)
        except:
            return None


def clean_data_chunk(chunk, value_column, timestamp_column):
    """Clean a chunk of data"""
    try:
        # Make copies of columns to avoid SettingWithCopyWarning
        chunk_clean = chunk.copy()

        # Convert value to numeric
        chunk_clean[value_column] = pd.to_numeric(chunk_clean[value_column], errors='coerce')

        # Convert timestamp to datetime
        chunk_clean[timestamp_column] = pd.to_datetime(chunk_clean[timestamp_column], errors='coerce')

        # Remove rows with missing critical values
        chunk_clean = chunk_clean.dropna(subset=[value_column, timestamp_column, 'id'])

        return chunk_clean

    except Exception as e:
        print(f"      ⚠️  Error cleaning chunk: {e}")
        return None


def aggregate_to_daily(chunk, value_column, timestamp_column):
    """Aggregate data to daily level"""
    try:
        # Extract date
        chunk['date'] = chunk[timestamp_column].dt.date

        # Group by id and date to get daily statistics
        daily_stats = chunk.groupby(['id', 'date'])[value_column].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()

        # Rename columns
        daily_stats = daily_stats.rename(columns={
            'mean': f'{value_column}_daily_mean',
            'std': f'{value_column}_daily_std',
            'min': f'{value_column}_daily_min',
            'max': f'{value_column}_daily_max',
            'count': f'{value_column}_daily_count'
        })

        return daily_stats

    except Exception as e:
        print(f"      ⚠️  Error in daily aggregation: {e}")
        return None


# =============================================================================
# PREPROCESSING AND EVALUATION
# =============================================================================

def preprocess_data(df):
    """Robust preprocessing that handles missing columns"""
    print("   ⚡ Preprocessing data...")

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


def evaluate_current_variables(df, current_best_variables=None):
    """Evaluate variables in current dataset"""
    print("   🎯 Evaluating variables...")

    # Get all potential predictors
    exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study', 'date']
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

    # Use correlation-based ranking
    variable_scores = []

    for variable in valid_predictors:
        scores = []
        for target in TARGETS:
            if target in df.columns:
                # Calculate correlation on complete cases
                valid_data = df[[variable, target]].dropna()
                if len(valid_data) > 10:
                    corr = valid_data[variable].corr(valid_data[target])
                    if not np.isnan(corr):
                        scores.append(abs(corr))

        if scores:
            avg_score = np.mean(scores)
            completeness = df[variable].notna().sum() / len(df)
            variable_scores.append((variable, avg_score, completeness))

    # Sort by weighted score (correlation * completeness)
    variable_scores.sort(key=lambda x: x[1] * x[2], reverse=True)

    # Take top variables from this dataset
    top_variables = [var for var, score, comp in variable_scores[:15]]

    print(f"   ✅ Selected top {len(top_variables)} variables")

    # Show top variables
    if top_variables:
        print("   Top variables from this file:")
        for i, (var, score, comp) in enumerate(variable_scores[:5]):
            source = "DAILY_AGG" if "daily" in var else "ORIGINAL"
            print(f"      {i + 1}. {var}: corr={score:.3f}, comp={comp:.1%} ({source})")

    # Combine with previous best variables
    if current_best_variables:
        combined_variables = list(set(current_best_variables + top_variables))
        print(f"   📈 Total variables so far: {len(combined_variables)}")
        return combined_variables
    else:
        return top_variables


def final_evaluation_with_test_set(df, best_variables):
    """Final evaluation with train/test split"""
    print(f"\n🏆 FINAL EVALUATION with {len(best_variables)} best variables")

    if not best_variables:
        print("   ⚠️  No variables to evaluate")
        return {}

    # Check which variables actually exist in the dataframe
    available_variables = [var for var in best_variables if var in df.columns]
    missing_variables = set(best_variables) - set(available_variables)

    if missing_variables:
        print(f"   ⚠️  {len(missing_variables)} variables not found in final data")
        print(f"      Missing: {list(missing_variables)[:5]}...")

    if len(available_variables) < 3:
        print("   ⚠️  Not enough variables for final evaluation")
        return {}

    # Prepare data
    X = df[available_variables].copy()
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

    print(f"   Train set: {X_train.shape}, Test set: {X_test.shape}")

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
    feature_importance = calculate_feature_importance(model, available_variables)

    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'target_results': target_results,
        'feature_importance': feature_importance,
        'n_features': len(available_variables),
        'n_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
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
    """Create final report"""
    print("\n📋 Creating final report...")

    if not results:
        print("   ⚠️  No results to report")
        return

    # Save results
    results['feature_importance'].to_csv(OUTPUT_DIR / 'final_feature_importance.csv', index=False)

    # Save variable list
    variable_df = pd.DataFrame({'variable': best_variables})
    variable_df.to_csv(OUTPUT_DIR / 'best_variables_list.csv', index=False)

    # Save processing history
    history_df = pd.DataFrame(processing_history)
    history_df.to_csv(OUTPUT_DIR / 'processing_history.csv', index=False)

    # Create report
    report = f"""
ROBUST SEQUENTIAL ANALYSIS REPORT
==================================

PROCESSING SUMMARY:
- Files processed: {len(processing_history)}
- Variables selected: {len(best_variables)}
- Features in final model: {results['n_features']}
- Total samples: {results['n_samples']}

FINAL MODEL PERFORMANCE:
- Overall Train R²: {results['train_r2']:.3f}
- Overall Test R²:  {results['test_r2']:.3f}

TARGET-SPECIFIC PERFORMANCE:
{chr(10).join([f'- {target}: Train R² = {results['target_results'][target]['train_r2']:.3f}, Test R² = {results['target_results'][target]['test_r2']:.3f}' for target in TARGETS])}

TOP 10 MOST IMPORTANT VARIABLES:
{chr(10).join([f'{i + 1:2d}. {row["variable"]:<40} : {row["importance_pct"]:5.2f}%' for i, row in results['feature_importance'].head(10).iterrows()])}

KEY IMPROVEMENTS:
1. Robust column detection and merging
2. Automatic handling of different file structures
3. Safe processing of high-frequency data
4. Graceful error handling for missing columns
"""

    with open(OUTPUT_DIR / 'robust_analysis_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("ROBUST SEQUENTIAL ANALYSIS")
    print("HANDLES FILES WITH DIFFERENT STRUCTURES")
    print("=" * 80)

    processing_history = []
    best_variables = []
    current_data = None

    try:
        # Step 1: Load base dataset
        current_data = load_base_dataset()
        current_data = preprocess_data(current_data)

        # Step 2: Process each file sequentially with robust handling
        for file_name in ALL_FILES:
            # Skip base file (already loaded)
            if file_name == 'hormones_and_selfreport.csv':
                best_variables = evaluate_current_variables(current_data)
                processing_history.append({
                    'file': file_name,
                    'n_variables': len(best_variables),
                    'data_shape': current_data.shape
                })
                continue

            # Process file with robust merging
            current_data = load_and_merge_single_file_robust(current_data, file_name)
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

        # Step 3: Final evaluation
        final_results = final_evaluation_with_test_set(current_data, best_variables)

        # Step 4: Create report
        if final_results:
            create_final_report(final_results, best_variables, processing_history)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📁 Results saved to: {OUTPUT_DIR}/")

        return final_results, best_variables

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, variables = main()