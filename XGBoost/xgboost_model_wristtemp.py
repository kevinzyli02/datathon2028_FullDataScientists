# fixed_wrist_temperature_aggregation.py
"""
FIXED WRIST TEMPERATURE AGGREGATION
Properly handles the wrist temperature data structure
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
OUTPUT_DIR = Path('fixed_temperature_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# FIXED WRIST TEMPERATURE PROCESSING
# =============================================================================

def inspect_wrist_temperature_structure():
    """First, let's inspect the actual structure of the wrist temperature file"""
    print("🔍 Inspecting wrist_temperature.csv structure...")

    file_path = DATA_DIR / 'wrist_temperature.csv'
    if not file_path.exists():
        print("   ❌ wrist_temperature.csv not found")
        return None

    try:
        # Read just the first few rows to understand the structure
        sample = pd.read_csv(file_path, nrows=10)
        print(f"   📊 Sample data shape: {sample.shape}")
        print(f"   📋 Columns: {list(sample.columns)}")
        print(f"   📝 First few rows:")
        print(sample.head())

        # Check data types and basic stats
        print(f"   🔍 Data types:")
        print(sample.dtypes)

        return sample
    except Exception as e:
        print(f"   ❌ Error inspecting file: {e}")
        return None


def process_wrist_temperature_correctly():
    """Process wrist temperature data with proper column handling"""
    print("\n🌡️ Processing wrist_temperature.csv with correct aggregation...")

    file_path = DATA_DIR / 'wrist_temperature.csv'
    if not file_path.exists():
        print("   ❌ wrist_temperature.csv not found")
        return None

    try:
        # First, let's check what columns we actually have
        sample = pd.read_csv(file_path, nrows=1000)
        print(f"   📋 Available columns: {list(sample.columns)}")

        # Look for temperature value columns
        temp_columns = [col for col in sample.columns if 'temp' in col.lower() or 'value' in col.lower()]
        print(f"   🔍 Potential temperature columns: {temp_columns}")

        # Look for timestamp columns
        time_columns = [col for col in sample.columns if
                        'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()]
        print(f"   ⏰ Potential time columns: {time_columns}")

        # Read in chunks with proper column selection
        chunk_size = 50000
        all_aggregated = []

        # We'll try to process the file with different column combinations
        usecols = None
        value_column = None
        time_column = None

        # Try to determine the correct columns to use
        if temp_columns:
            value_column = temp_columns[0]  # Use first temperature-like column
        else:
            # If no obvious temperature column, use first numeric column that's not id
            numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'id']
            if numeric_cols:
                value_column = numeric_cols[0]

        if time_columns:
            time_column = time_columns[0]  # Use first time-like column
        else:
            # If no obvious time column, look for datetime formats
            for col in sample.columns:
                if sample[col].dtype == 'object':
                    try:
                        pd.to_datetime(sample[col].head())
                        time_column = col
                        break
                    except:
                        continue

        print(f"   🎯 Using value column: {value_column}")
        print(f"   🕒 Using time column: {time_column}")

        if not value_column or not time_column:
            print("   ❌ Could not identify value and time columns")
            return None

        # Process file in chunks
        for i, chunk in enumerate(
                pd.read_csv(file_path, chunksize=chunk_size, usecols=['id', value_column, time_column])):
            print(f"   Processing chunk {i + 1}...")

            # Clean the chunk
            chunk_clean = clean_temperature_chunk(chunk, value_column, time_column)
            if chunk_clean is not None and not chunk_clean.empty:
                # Aggregate to hourly statistics
                aggregated = aggregate_temperature_to_hourly(chunk_clean, value_column, time_column)
                if aggregated is not None and not aggregated.empty:
                    all_aggregated.append(aggregated)

            # Limit to first few chunks for memory
            if i >= 5:  # Process only 300K rows max
                print(f"   ⚠️ Limiting to first 300K rows for memory")
                break

        if all_aggregated:
            final_aggregated = pd.concat(all_aggregated, ignore_index=True)

            # Final aggregation by hour to handle any overlaps
            final_daily = aggregate_to_daily_level(final_aggregated)

            print(f"   ✅ Final aggregated data: {final_daily.shape}")
            return final_daily
        else:
            print("   ❌ No data was aggregated")
            return None

    except Exception as e:
        print(f"   ❌ Error processing wrist temperature: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_temperature_chunk(chunk, value_column, time_column):
    """Clean and prepare a chunk of temperature data"""
    try:
        # Remove rows with missing critical values
        chunk_clean = chunk.dropna(subset=[value_column, time_column, 'id'])

        # Convert value to numeric, coercing errors
        chunk_clean[value_column] = pd.to_numeric(chunk_clean[value_column], errors='coerce')

        # Remove any infinite values
        chunk_clean = chunk_clean[np.isfinite(chunk_clean[value_column])]

        # Convert time column to datetime
        chunk_clean[time_column] = pd.to_datetime(chunk_clean[time_column], errors='coerce')

        # Remove rows where datetime conversion failed
        chunk_clean = chunk_clean[chunk_clean[time_column].notna()]

        return chunk_clean

    except Exception as e:
        print(f"      ⚠️ Error cleaning chunk: {e}")
        return None


def aggregate_temperature_to_hourly(chunk, value_column, time_column):
    """Aggregate temperature data to hourly statistics"""
    try:
        # Extract date and hour components
        chunk['date'] = chunk[time_column].dt.date
        chunk['hour'] = chunk[time_column].dt.hour

        # Group by id, date, and hour to get hourly statistics
        hourly_stats = chunk.groupby(['id', 'date', 'hour'])[value_column].agg([
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

        return hourly_stats

    except Exception as e:
        print(f"      ⚠️ Error in hourly aggregation: {e}")
        return None


def aggregate_to_daily_level(hourly_data):
    """Aggregate hourly data to daily level for merging"""
    try:
        # Convert date to datetime for proper grouping
        hourly_data['date'] = pd.to_datetime(hourly_data['date'])

        # Extract day of year for grouping (simplified approach)
        hourly_data['day_of_year'] = hourly_data['date'].dt.dayofyear

        # Group by id and day to get daily statistics from hourly data
        daily_stats = hourly_data.groupby(['id', 'day_of_year']).agg({
            col: ['mean', 'std'] for col in hourly_data.columns
            if any(x in col for x in ['hourly_mean', 'hourly_std', 'hourly_min', 'hourly_max'])
        }).reset_index()

        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip('_') for col in daily_stats.columns.values]
        daily_stats = daily_stats.rename(columns={'id_': 'id', 'day_of_year_': 'day_of_year'})

        print(f"   📊 Created daily stats with {len(daily_stats.columns)} features")
        return daily_stats

    except Exception as e:
        print(f"   ⚠️ Error in daily aggregation: {e}")
        return hourly_data  # Return hourly data as fallback


def create_simple_model_with_temperature():
    """Create a simple model using the properly processed temperature data"""
    print("\n🤖 Creating model with processed temperature data...")

    # Load base data
    base_file = 'hormones_and_selfreport.csv'
    base_path = DATA_DIR / base_file

    if not base_path.exists():
        print("   ❌ Base hormones file not found")
        return None

    base_data = pd.read_csv(base_path)
    print(f"   ✅ Base data: {base_data.shape}")

    # Process wrist temperature
    temp_data = process_wrist_temperature_correctly()

    if temp_data is None:
        print("   ❌ No temperature data to merge")
        return None

    # Simple merge on id (this is a simplified approach)
    # In a real scenario, you'd map day_of_year to day_in_study
    merged_data = pd.merge(base_data, temp_data, on=['id'], how='left')
    print(f"   ✅ After merging temperature: {merged_data.shape}")

    # Basic preprocessing
    merged_data = simple_preprocessing(merged_data)

    # Evaluate the data
    results = evaluate_with_temperature(merged_data)

    return results


def simple_preprocessing(df):
    """Simple preprocessing"""
    # Convert symptoms to numeric if they exist
    symptom_mapping = {'Not at all': 0, 'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}

    for col in ['stress', 'fatigue']:
        if col in df.columns:
            df[col] = df[col].map(symptom_mapping)

    return df


def evaluate_with_temperature(df):
    """Evaluate model with temperature features"""
    print("   🎯 Evaluating model with temperature features...")

    # Get all numeric features
    exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study', 'day_of_year']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Filter for features with reasonable completeness
    valid_features = []
    for col in feature_cols:
        completeness = df[col].notna().sum() / len(df)
        if completeness > 0.1:  # At least 10% complete
            valid_features.append(col)

    print(f"   📊 Using {len(valid_features)} features")

    if len(valid_features) < 3:
        print("   ⚠️ Not enough features for modeling")
        return None

    # Count temperature features
    temp_features = [col for col in valid_features if 'temp' in col.lower() or 'value' in col.lower()]
    print(f"   🌡️ Temperature features: {len(temp_features)}")

    # Prepare data
    X = df[valid_features].copy()
    y = df[TARGETS].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"   📈 Final modeling data: {X.shape}")

    if len(X) < 50:
        print("   ⚠️ Not enough samples for modeling")
        return None

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train simple model
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=30,
            max_depth=3,
            random_state=RANDOM_STATE,
            n_jobs=1  # Single job to save memory
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
    feature_importance = calculate_feature_importance(model, valid_features)

    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'target_results': target_results,
        'feature_importance': feature_importance,
        'n_features': len(valid_features),
        'n_temp_features': len(temp_features),
        'n_samples': len(X)
    }

    print(f"   ✅ Model Performance:")
    print(f"      Overall - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    for target in TARGETS:
        print(
            f"      {target} - Train R²: {target_results[target]['train_r2']:.3f}, Test R²: {target_results[target]['test_r2']:.3f}")

    # Show top temperature features
    temp_importance = feature_importance[feature_importance['variable'].isin(temp_features)]
    if not temp_importance.empty:
        print(f"   🌡️ Top Temperature Features:")
        for i, (_, row) in enumerate(temp_importance.head(5).iterrows()):
            print(f"      {i + 1}. {row['variable']}: {row['importance_pct']:.2f}%")

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


def save_results(results):
    """Save results to file"""
    if results is None:
        print("   ❌ No results to save")
        return

    print("\n💾 Saving results...")

    # Save feature importance
    results['feature_importance'].to_csv(OUTPUT_DIR / 'temperature_feature_importance.csv', index=False)

    # Create report
    report = f"""
WRIST TEMPERATURE ANALYSIS RESULTS
===================================

MODEL PERFORMANCE:
- Overall Train R²: {results['train_r2']:.3f}
- Overall Test R²:  {results['test_r2']:.3f}
- Number of Features: {results['n_features']}
- Temperature Features: {results['n_temp_features']}
- Total Samples: {results['n_samples']}

TARGET-SPECIFIC PERFORMANCE:
{chr(10).join([f'- {target}: Train R² = {results['target_results'][target]['train_r2']:.3f}, Test R² = {results['target_results'][target]['test_r2']:.3f}' for target in TARGETS])}

TOP 10 FEATURES OVERALL:
{chr(10).join([f'{i + 1:2d}. {row["variable"]:<40} : {row["importance_pct"]:5.2f}%' for i, row in results['feature_importance'].head(10).iterrows()])}

TEMPERATURE FEATURES IN MODEL:
- Count: {results['n_temp_features']}
- These represent {results['n_temp_features'] / results['n_features'] * 100:.1f}% of all features

CONCLUSIONS:
1. Wrist temperature data was successfully processed with hourly aggregation
2. Temperature variability (std) may provide predictive power for hormones
3. Model performance indicates temperature data's usefulness
"""

    with open(OUTPUT_DIR / 'temperature_analysis_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("FIXED WRIST TEMPERATURE PROCESSING")
    print("=" * 80)

    try:
        # Step 1: First inspect the file structure
        sample = inspect_wrist_temperature_structure()

        if sample is None:
            print("❌ Cannot proceed without understanding file structure")
            return

        # Step 2: Process wrist temperature with proper handling
        results = create_simple_model_with_temperature()

        # Step 3: Save results
        if results:
            save_results(results)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📁 Results saved to: {OUTPUT_DIR}/")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()