# sequential_analysis_with_hourly_aggregation.py
"""
SEQUENTIAL ANALYSIS WITH HOURLY AGGREGATION
Handles high-frequency data by aggregating to hourly statistics
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
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('hourly_aggregation_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of files to process sequentially
ALL_FILES = [
    'hormones_and_selfreport.csv',  # Base file first

    # Original key files
    #'active_minutes.csv',
    'sleep.csv',
    'stress_score.csv',
    #'steps.csv',
    'resting_heart_rate.csv',
    'glucose.csv',
    'heart_rate_variability_details.csv',
    'computed_temperature.csv',
    #'demographic_vo2_max.csv',
    'height_and_weight.csv',
    #'subject-info.csv',

    # NEWLY REQUESTED FILES
    #'active_zone_minutes.csv',
    #'calories.csv',
    #'estimated_oxygen_variation.csv',
    'exercise.csv',
    'respiratory_rate_summary.csv',
    'sleep_score.csv',
    #'wrist_temperature.csv'
    ]

TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# HOURLY AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_high_frequency_data(df, value_column='value', timestamp_column='timestamp'):
    """
    Aggregate high-frequency data (minute-level) to hourly statistics
    Returns hourly mean and standard deviation
    """
    print(f"   📊 Aggregating high-frequency data from {len(df):,} records...")

    # Check if we have the required columns
    if timestamp_column not in df.columns:
        print(f"   ⚠️  No {timestamp_column} column found, skipping aggregation")
        return df

    if value_column not in df.columns:
        # Try to find a numeric column to aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_column = numeric_cols[0]
            print(f"   Using {value_column} as value column for aggregation")
        else:
            print("   ⚠️  No numeric columns found for aggregation")
            return df

    try:
        # Convert timestamp to datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Extract date and hour for grouping
        df['date'] = df[timestamp_column].dt.date
        df['hour'] = df[timestamp_column].dt.hour

        # Group by id, date, and hour to get hourly statistics
        hourly_stats = df.groupby(['id', 'date', 'hour'])[value_column].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()

        # Rename columns for clarity
        hourly_stats = hourly_stats.rename(columns={
            'mean': f'{value_column}_hourly_mean',
            'std': f'{value_column}_hourly_std',
            'min': f'{value_column}_hourly_min',
            'max': f'{value_column}_hourly_max',
            'count': f'{value_column}_hourly_count'
        })

        print(f"   ✅ Aggregated to {len(hourly_stats):,} hourly records")
        return hourly_stats

    except Exception as e:
        print(f"   ❌ Error in hourly aggregation: {e}")
        return df


def map_hourly_to_day_in_study(hourly_data, reference_data):
    """
    Map hourly data to day_in_study using timestamp information
    """
    print("   🔄 Mapping hourly data to day_in_study...")

    # This is a simplified mapping - you might need to adjust based on your data structure
    # For now, we'll aggregate hourly data to daily statistics
    daily_stats = hourly_data.groupby('id').agg({
        col: ['mean', 'std'] for col in hourly_data.columns
        if any(x in col for x in ['hourly_mean', 'hourly_std', 'hourly_min', 'hourly_max'])
    }).reset_index()

    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip('_') for col in daily_stats.columns.values]
    daily_stats = daily_stats.rename(columns={'id_': 'id'})

    print(f"   ✅ Created daily statistics with {len(daily_stats.columns)} features")
    return daily_stats


def process_high_frequency_file(file_path, file_name):
    """
    Process high-frequency files with special aggregation
    """
    print(f"   🔍 Processing high-frequency file: {file_name}")

    try:
        # Read only first few rows to check structure
        sample = pd.read_csv(file_path, nrows=1000)
        print(f"   📋 File structure: {sample.shape}, columns: {list(sample.columns)}")

        # Check if this is a high-frequency file that needs aggregation
        high_freq_indicators = ['timestamp', 'datetime', 'time', 'minute', 'second']
        is_high_frequency = any(indicator in str(sample.columns).lower() for indicator in high_freq_indicators)

        if is_high_frequency and len(sample) == 1000:
            print(f"   ⚡ High-frequency file detected, reading in chunks...")

            # Read file in chunks for memory efficiency
            chunk_size = 100000
            chunks = []

            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                print(f"      Processing chunk {i + 1}...")
                aggregated_chunk = aggregate_high_frequency_data(chunk)
                if aggregated_chunk is not None and not aggregated_chunk.empty:
                    chunks.append(aggregated_chunk)

                # Limit chunks for very large files to avoid memory issues
                if i >= 10:  # Process only first 1M rows (10 chunks)
                    print(f"      ⚠️  Limiting to first 1M rows for memory efficiency")
                    break

            if chunks:
                aggregated_data = pd.concat(chunks, ignore_index=True)
                print(f"   ✅ Aggregated data: {aggregated_data.shape}")
                return aggregated_data
            else:
                print(f"   ⚠️  No data aggregated from chunks")
                return sample  # Return sample as fallback
        else:
            # Regular file, read normally
            df = pd.read_csv(file_path)
            print(f"   ✅ Regular file loaded: {df.shape}")
            return df

    except Exception as e:
        print(f"   ❌ Error processing {file_name}: {e}")
        return None


# =============================================================================
# UPDATED SEQUENTIAL PROCESSING FUNCTIONS
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
    """Load and merge a single file with the current base data - UPDATED FOR HIGH-FREQUENCY"""
    print(f"\n🔄 Processing {file_name}...")

    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"   ❌ {file_name}: Not found")
        return base_data

    try:
        # Special handling for high-frequency files
        high_frequency_files = ['wrist_temperature.csv', 'estimated_oxygen_variation.csv']

        if file_name in high_frequency_files:
            new_data = process_high_frequency_file(file_path, file_name)
        else:
            new_data = pd.read_csv(file_path)

        if new_data is None:
            print(f"   ⚠️  Could not load {file_name}")
            return base_data

        print(f"   ✅ {file_name}: {new_data.shape}")

        # Define merge strategy for this file
        if file_name == 'hormones_and_selfreport.csv':
            return base_data  # Already loaded as base

        elif file_name in ['sleep.csv', 'respiratory_rate_summary.csv', 'sleep_score.csv']:
            # Sleep-related data uses sleep_start_day_in_study
            if 'sleep_start_day_in_study' in new_data.columns:
                new_data = new_data.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
            merged_data = pd.merge(base_data, new_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name in ['demographic_vo2_max.csv', 'height_and_weight.csv', 'subject-info.csv']:
            # Demographic data - merge on id only
            merged_data = pd.merge(base_data, new_data, on=['id'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name in ['wrist_temperature.csv', 'estimated_oxygen_variation.csv']:
            # High-frequency data - already aggregated, merge on id only for now
            # In a more sophisticated version, you'd map timestamps to day_in_study
            if 'id' in new_data.columns:
                merged_data = pd.merge(base_data, new_data, on=['id'], how='left',
                                       suffixes=('', f'_{file_name.replace(".csv", "")}'))
            else:
                print(f"   ⚠️  No id column in aggregated data, skipping merge")
                return base_data

        elif file_name == 'heart_rate_variability_details.csv':
            # Aggregate HRV data first
            aggregated_data = aggregate_hrv_data_simple(new_data)
            merged_data = pd.merge(base_data, aggregated_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name == 'exercise.csv':
            # Aggregate exercise data
            aggregated_data = aggregate_exercise_data(new_data)
            merged_data = pd.merge(base_data, aggregated_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        elif file_name == 'active_zone_minutes.csv':
            # Active zone minutes might need aggregation
            aggregated_data = aggregate_active_zone_minutes(new_data)
            merged_data = pd.merge(base_data, aggregated_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        else:
            # Default merge on id and day_in_study for daily metrics
            merged_data = pd.merge(base_data, new_data, on=['id', 'day_in_study'], how='left',
                                   suffixes=('', f'_{file_name.replace(".csv", "")}'))

        print(f"   ✅ After {file_name}: {merged_data.shape}")
        return merged_data

    except Exception as e:
        print(f"   ❌ Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
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


def aggregate_exercise_data(df):
    """Aggregate exercise data"""
    if 'start_day_in_study' not in df.columns:
        return df

    exercise_metrics = ['duration', 'calories', 'averageheartrate', 'steps']
    available_metrics = [col for col in exercise_metrics if col in df.columns]

    if not available_metrics:
        return df

    # Aggregate exercise data
    aggregation = {col: ['sum', 'mean', 'max'] for col in available_metrics}
    aggregated = df.groupby(['id', 'start_day_in_study']).agg(aggregation).reset_index()

    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    aggregated = aggregated.rename(columns={'start_day_in_study': 'day_in_study'})

    return aggregated


def aggregate_active_zone_minutes(df):
    """Aggregate active zone minutes data"""
    if 'day_in_study' not in df.columns:
        return df

    zone_metrics = ['fat_burn_minutes', 'cardio_minutes', 'peak_minutes']
    available_metrics = [col for col in zone_metrics if col in df.columns]

    if not available_metrics:
        return df

    # Sum minutes by zone
    aggregated = df.groupby(['id', 'day_in_study'])[available_metrics].sum().reset_index()

    return aggregated


# =============================================================================
# PREPROCESSING AND EVALUATION (UNCHANGED)
# =============================================================================

def preprocess_data(df):
    """Fast preprocessing for the current dataset"""
    print("   ⚡ Preprocessing data...")

    # Convert symptoms to numeric
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

    # Handle categorical variables quickly
    categorical_cols = [
        'phase', 'flow_volume', 'flow_color', 'gender', 'ethnicity',
        'education', 'employment', 'income', 'sexually_active',
        'self_report_menstrual_health_literacy'
    ]

    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            try:
                df[col] = df[col].astype('category').cat.codes
                df[col] = df[col].replace(-1, np.nan)
            except:
                pass  # Skip if conversion fails

    return df


def create_derived_features(df):
    """Create derived features from the accumulated data"""
    print("   🔄 Creating derived features...")

    # Activity features
    if all(col in df.columns for col in ['lightly', 'moderately', 'very']):
        df['total_activity_minutes'] = df['lightly'] + df['moderately'] + df['very']
        df['intense_activity_ratio'] = df['very'] / (df['lightly'] + 1)
        print("      Created activity features")

    # Sleep features
    if all(col in df.columns for col in ['minutesasleep', 'timeinbed']):
        df['sleep_efficiency'] = df['minutesasleep'] / (df['timeinbed'] + 1)
        print("      Created sleep efficiency")

    # Temperature variability features (from hourly aggregation)
    temp_cols = [col for col in df.columns if 'temperature' in col and 'hourly_std' in col]
    for col in temp_cols:
        base_name = col.replace('_hourly_std', '')
        mean_col = f'{base_name}_hourly_mean'
        if mean_col in df.columns:
            df[f'{base_name}_cv'] = df[col] / (df[mean_col] + 1)  # Coefficient of variation
            print(f"      Created {base_name}_cv (variability measure)")

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

    # Use correlation-based ranking for speed
    variable_scores = []

    for variable in valid_predictors:
        scores = []
        for target_idx, target in enumerate(TARGETS):
            # Calculate correlation on complete cases
            valid_data = pd.concat([df[variable], df[target]], axis=1).dropna()
            if len(valid_data) > 10:
                corr = valid_data[variable].corr(valid_data[target])
                if not np.isnan(corr):
                    scores.append(abs(corr))

        if scores:
            avg_score = np.mean(scores)
            completeness = df[variable].notna().sum() / len(df)
            variable_scores.append((variable, avg_score, completeness))

    # Sort by average correlation (weighted by completeness)
    variable_scores.sort(key=lambda x: x[1] * x[2], reverse=True)  # Correlation * completeness

    # Take top 15 variables from this dataset
    top_variables = [var for var, score, comp in variable_scores[:15]]

    print(f"   ✅ Selected top {len(top_variables)} variables")

    # Show top 5 from this file
    if top_variables:
        print("   Top variables from this file:")
        for i, (var, score, comp) in enumerate(variable_scores[:5]):
            source = "HOURLY_AGG" if "hourly" in var else "DAILY"
            print(f"      {i + 1}. {var}: corr={score:.3f}, comp={comp:.1%} ({source})")

    # Combine with previous best variables
    if current_best_variables:
        combined_variables = list(set(current_best_variables + top_variables))
        print(f"   📈 Total variables so far: {len(combined_variables)}")
        return combined_variables
    else:
        return top_variables


# =============================================================================
# FINAL EVALUATION AND REPORTING (UNCHANGED)
# =============================================================================

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
    feature_importance = calculate_feature_importance(model, best_variables)

    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'target_results': target_results,
        'feature_importance': feature_importance,
        'n_features': len(best_variables),
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
    """Create final report and visualizations"""
    print("\n📋 Creating final report...")

    if not results:
        print("   ⚠️  No results to report")
        return

    # Save detailed results
    results['feature_importance'].to_csv(OUTPUT_DIR / 'final_feature_importance.csv', index=False)

    # Save variable list
    variable_df = pd.DataFrame({'variable': best_variables})
    variable_df.to_csv(OUTPUT_DIR / 'best_variables_list.csv', index=False)

    # Save processing history
    history_df = pd.DataFrame(processing_history)
    history_df.to_csv(OUTPUT_DIR / 'processing_history.csv', index=False)

    # Create comprehensive summary report
    report = f"""
HOURLY AGGREGATION ANALYSIS REPORT
====================================

PROCESSING SUMMARY:
- Files processed: {len(processing_history)}
- Final variables selected: {len(best_variables)}
- Total samples: {results['n_samples']}
- Train samples: {results['train_samples']}
- Test samples: {results['test_samples']}

FINAL MODEL PERFORMANCE:
- Overall Train R²: {results['train_r2']:.3f}
- Overall Test R²:  {results['test_r2']:.3f}

TARGET-SPECIFIC PERFORMANCE:
{chr(10).join([f'- {target}: Train R² = {results['target_results'][target]['train_r2']:.3f}, Test R² = {results['target_results'][target]['test_r2']:.3f}' for target in TARGETS])}

TOP 10 MOST IMPORTANT VARIABLES:
{chr(10).join([f'{i + 1:2d}. {row["variable"]:<40} : {row["importance_pct"]:5.2f}%' for i, row in results['feature_importance'].head(10).iterrows()])}

HOURLY AGGREGATION FEATURES:
- Count of hourly mean features: {len([v for v in best_variables if 'hourly_mean' in v])}
- Count of hourly std features: {len([v for v in best_variables if 'hourly_std' in v])}
- Count of hourly variability (CV) features: {len([v for v in best_variables if '_cv' in v])}

PROCESSING HISTORY:
{chr(10).join([f'- {entry["file"]}: {entry["n_variables"]} variables' for entry in processing_history])}

KEY IMPROVEMENTS:
1. High-frequency data (wrist_temperature, etc.) aggregated to hourly statistics
2. Memory-efficient chunk processing for large files
3. Hourly mean, std, min, max, and count features created
4. Coefficient of variation (CV) for temperature variability analysis
"""

    with open(OUTPUT_DIR / 'hourly_aggregation_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return report


# =============================================================================
# MAIN SEQUENTIAL PROCESSING
# =============================================================================

def main():
    """Main sequential processing function with hourly aggregation"""
    print("=" * 80)
    print("SEQUENTIAL ANALYSIS WITH HOURLY AGGREGATION")
    print("HANDLES HIGH-FREQUENCY DATA EFFICIENTLY")
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

            # Create derived features after significant merges
            if file_name in ['wrist_temperature.csv', 'estimated_oxygen_variation.csv']:
                current_data = create_derived_features(current_data)

            # Evaluate current variables
            best_variables = evaluate_current_variables(current_data, best_variables)

            # Record processing step
            processing_history.append({
                'file': file_name,
                'n_variables': len(best_variables),
                'data_shape': current_data.shape
            })

            # Clean memory between files
            gc.collect()

        print(f"\n✅ Sequential processing completed!")
        print(f"   Total variables accumulated: {len(best_variables)}")
        print(f"   Final dataset shape: {current_data.shape}")

        # Count hourly aggregation features
        hourly_features = [v for v in best_variables if 'hourly' in v]
        print(f"   📊 Hourly aggregation features: {len(hourly_features)}")

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
        print("📊 Hourly aggregation successfully handled high-frequency data")

        return final_results, best_variables

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, variables = main()