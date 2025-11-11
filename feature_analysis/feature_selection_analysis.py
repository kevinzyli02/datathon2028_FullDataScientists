# feature_selection_analysis_chunked.py
"""
COMPLETE FEATURE SELECTION ANALYSIS WITH CHUNKED PROCESSING
Uses your working directory paths with memory-efficient chunking
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import warnings
import os
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - USING YOUR WORKING PATHS
# =============================================================================

DATA_DIR = Path(t
    r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")
OUTPUT_DIR = Path('optimized_sequential_analysis')
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
                   #'wrist_temperature.csv'
                   ]

TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Memory optimization parameters
CHUNK_SIZE = 10000  # Process this many rows at a time
MAX_FEATURES_PER_CHUNK = 50  # Process this many features at a time
MEMORY_EFFICIENT_MERGE = True

# Feature selection parameters
RFE_N_FEATURES = 15
RF_N_FEATURES = 15


# =============================================================================
# MEMORY-EFFICIENT DATA PROCESSING FUNCTIONS
# =============================================================================

def load_base_dataset_chunked():
    """Load base dataset with memory optimization"""
    print("📂 Loading base dataset...")
    base_file = 'hormones_and_selfreport.csv'
    file_path = DATA_DIR / base_file

    if not file_path.exists():
        raise ValueError(f"Base file {base_file} not found")

    # Load with optimized dtypes
    base_data = pd.read_csv(file_path)

    # Optimize memory usage
    base_data = optimize_dataframe_memory(base_data)

    print(f"   ✅ Base data: {base_data.shape}")
    print(f"   💾 Memory usage: {base_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    return base_data


def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage by downcasting numeric types"""
    print("   🔧 Optimizing memory usage...")

    # Downcast numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert object columns to category if they have few unique values
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')

    return df


def load_and_merge_single_file_chunked(base_data, file_name):
    """Memory-efficient file loading and merging with chunking"""
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

        # Load the file with memory optimization
        new_data = pd.read_csv(file_path)
        new_data = optimize_dataframe_memory(new_data)

        print(f"   ✅ {file_name}: {new_data.shape}")
        print(f"   💾 New data memory: {new_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Find merge columns for this file
        merge_columns = find_merge_columns(new_data, file_name)

        if merge_columns is None:
            print(f"   ⚠️  Cannot merge {file_name} - no suitable columns found")
            return base_data

        # Handle special cases
        if file_name == 'hormones_and_selfreport.csv':
            return base_data

        # Apply special case handling
        new_data = handle_special_cases(new_data, file_name, merge_columns)

        # Check if all merge columns exist
        missing_in_base = [col for col in merge_columns if col not in base_data.columns]
        missing_in_new = [col for col in merge_columns if col not in new_data.columns]

        if missing_in_base:
            print(f"   ⚠️  Merge columns {missing_in_base} missing in base data")
            return base_data

        if missing_in_new:
            print(f"   ⚠️  Merge columns {missing_in_new} missing in {file_name}")
            return base_data

        # MEMORY-EFFICIENT MERGE: Select only necessary columns
        print(f"   🔗 Merging on columns: {merge_columns}")

        # Get base data columns to keep (targets + merge keys + existing features)
        base_keep_cols = list(set(merge_columns + TARGETS + list(base_data.columns)))
        base_keep_cols = [col for col in base_keep_cols if col in base_data.columns]

        # Get new data columns to keep (merge keys + new features)
        new_keep_cols = list(set(merge_columns + [col for col in new_data.columns if col not in base_data.columns]))
        new_keep_cols = [col for col in new_keep_cols if col in new_data.columns]

        # Create optimized subsets
        base_subset = base_data[base_keep_cols].copy()
        new_subset = new_data[new_keep_cols].copy()

        print(f"   📊 Base subset: {base_subset.shape}, New subset: {new_subset.shape}")

        # Perform the merge on subsets
        merged_data = pd.merge(
            base_subset,
            new_subset,
            on=merge_columns,
            how='left',
            suffixes=('', f'_{file_name.replace(".csv", "")}')
        )

        print(f"   ✅ After {file_name}: {merged_data.shape}")
        print(f"   💾 Merged memory: {merged_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Clean up memory
        del base_subset, new_subset
        gc.collect()

        return merged_data

    except MemoryError as e:
        print(f"   💥 Memory error processing {file_name}: {e}")
        print(f"   🔄 Falling back to chunked merge...")
        return merge_large_file_chunked(base_data, file_name, merge_columns)
    except Exception as e:
        print(f"   ❌ Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return base_data


def merge_large_file_chunked(base_data, file_name, merge_columns):
    """Merge very large files using chunking"""
    print(f"   🔄 Using chunked merge for {file_name}...")

    file_path = DATA_DIR / file_name

    try:
        # Get unique IDs from base data for chunking
        if 'id' in base_data.columns:
            unique_ids = base_data['id'].unique()
            chunks = np.array_split(unique_ids, max(1, len(unique_ids) // 100))  # ~100 IDs per chunk

            merged_chunks = []

            for i, id_chunk in enumerate(chunks):
                print(f"      Processing chunk {i + 1}/{len(chunks)}...")

                # Filter base data for this chunk
                base_chunk = base_data[base_data['id'].isin(id_chunk)].copy()

                # Load and filter new data for this chunk
                new_chunk = pd.read_csv(file_path)
                new_chunk = new_chunk[new_chunk['id'].isin(id_chunk)]

                if not new_chunk.empty:
                    # Merge this chunk
                    merged_chunk = pd.merge(
                        base_chunk,
                        new_chunk,
                        on=merge_columns,
                        how='left',
                        suffixes=('', f'_{file_name.replace(".csv", "")}')
                    )
                    merged_chunks.append(merged_chunk)

                # Clean memory
                del base_chunk, new_chunk
                gc.collect()

            # Combine all chunks
            if merged_chunks:
                merged_data = pd.concat(merged_chunks, ignore_index=True)
                print(f"   ✅ Chunked merge completed: {merged_data.shape}")
                return merged_data
            else:
                print(f"   ⚠️  No data merged in chunked approach")
                return base_data

        else:
            print(f"   ⚠️  Cannot use chunked merge - no 'id' column")
            return base_data

    except Exception as e:
        print(f"   ❌ Error in chunked merge: {e}")
        return base_data


def handle_special_cases(new_data, file_name, merge_columns):
    """Handle special cases for different file types"""
    if file_name in ['sleep.csv', 'respiratory_rate_summary.csv', 'sleep_score.csv', 'computed_temperature.csv']:
        if 'sleep_start_day_in_study' in new_data.columns and 'day_in_study' not in new_data.columns:
            new_data = new_data.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
            if 'sleep_start_day_in_study' in merge_columns:
                merge_columns = ['id', 'day_in_study']

    elif file_name == 'exercise.csv':
        if 'start_day_in_study' in new_data.columns and 'day_in_study' not in new_data.columns:
            new_data = new_data.rename(columns={'start_day_in_study': 'day_in_study'})
            if 'start_day_in_study' in merge_columns:
                merge_columns = ['id', 'day_in_study']

    return new_data


def inspect_file_columns(file_name):
    """Inspect a file's columns to understand its structure"""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"   ❌ {file_name}: File not found")
        return None

    try:
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


def evaluate_current_variables_chunked(df, current_best_variables=None):
    """Evaluate variables in chunks to prevent memory issues"""
    print("   🎯 Evaluating variables in chunks...")

    # Get all potential predictors
    exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study', 'date']
    all_predictors = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    print(f"   📊 Total variables to evaluate: {len(all_predictors)}")

    # Process in chunks
    chunk_size = MAX_FEATURES_PER_CHUNK
    variable_scores = []

    for i in range(0, len(all_predictors), chunk_size):
        chunk_predictors = all_predictors[i:i + chunk_size]
        print(
            f"   🔄 Processing chunk {i // chunk_size + 1}/{(len(all_predictors) - 1) // chunk_size + 1} ({len(chunk_predictors)} variables)")

        chunk_scores = evaluate_variables_chunk(df, chunk_predictors)
        variable_scores.extend(chunk_scores)

        # Clean memory between chunks
        gc.collect()

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


def evaluate_variables_chunk(df, predictors_chunk):
    """Evaluate a chunk of variables"""
    variable_scores = []

    for variable in predictors_chunk:
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

    return variable_scores


# =============================================================================
# DATA PREPARATION FOR FEATURE SELECTION
# =============================================================================

def create_processed_dataset_chunked():
    """Create the processed dataset using memory-efficient chunked processing"""
    print("🚀 Creating processed dataset with chunked processing...")

    processing_history = []
    best_variables = []
    current_data = None

    try:
        # Step 1: Load base dataset
        current_data = load_base_dataset_chunked()
        current_data = preprocess_data(current_data)

        # Step 2: Process each file sequentially with chunking
        for file_name in OPTIMIZED_FILES:
            if file_name == 'hormones_and_selfreport.csv':
                best_variables = evaluate_current_variables_chunked(current_data)
                processing_history.append({
                    'file': file_name,
                    'n_variables': len(best_variables),
                    'data_shape': current_data.shape
                })
                continue

            print(f"\n📊 Current memory before {file_name}:")
            print(f"   💾 Memory usage: {current_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

            current_data = load_and_merge_single_file_chunked(current_data, file_name)
            current_data = preprocess_data(current_data)

            best_variables = evaluate_current_variables_chunked(current_data, best_variables)

            processing_history.append({
                'file': file_name,
                'n_variables': len(best_variables),
                'data_shape': current_data.shape
            })

            # Aggressive memory cleanup
            gc.collect()

        print(f"\n✅ Dataset processing completed!")
        print(f"   Total variables accumulated: {len(best_variables)}")
        print(f"   Final dataset: {current_data.shape}")

        # Save the processed dataset
        current_data.to_csv(OUTPUT_DIR / 'merged_dataset.csv', index=False)

        # Save candidate variables
        variable_df = pd.DataFrame({'variable': best_variables})
        variable_df.to_csv(OUTPUT_DIR / 'candidate_variables.csv', index=False)

        print(f"💾 Data saved to: {OUTPUT_DIR}/")

        return current_data, best_variables

    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# MEMORY-EFFICIENT FEATURE SELECTION METHODS
# =============================================================================

def rfe_feature_selection_chunked(X, y, n_features=RFE_N_FEATURES):
    """Recursive Feature Elimination with chunked processing"""
    print(f"\n🔄 Performing RFE to select {n_features} features...")

    if len(X) < 50:
        print(f"⚠️  Not enough data for RFE ({len(X)} samples)")
        return list(X.columns), pd.DataFrame(), {}

    # Use only a subset of data for RFE to save memory
    if len(X) > 10000:
        print(f"   📉 Subsampling to 10,000 samples for memory efficiency")
        X_sample = X.sample(n=10000, random_state=RANDOM_STATE)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_train.columns,
        index=X_test.index
    )

    # XGBoost estimator for RFE with reduced parameters for memory
    xgb_estimator = XGBRegressor(
        n_estimators=50,  # Reduced for memory
        max_depth=4,  # Reduced for memory
        random_state=RANDOM_STATE,
        n_jobs=1  # Single job to reduce memory
    )

    # RFE for each target separately
    all_selected_features = []
    feature_rankings = []
    performance_results = {}

    for i, target in enumerate(TARGETS):
        print(f"   🎯 RFE for target: {target}")

        # Single target RFE
        rfe = RFE(
            estimator=xgb_estimator,
            n_features_to_select=n_features,
            step=max(1, int(len(X.columns) * 0.2))  # Larger steps for memory
        )

        rfe.fit(X_train_imputed, y_train.iloc[:, i])

        # Get selected features for this target
        target_features = X.columns[rfe.support_].tolist()
        all_selected_features.extend(target_features)

        # Store ranking
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        }).sort_values('ranking')

        ranking_df['target'] = target
        feature_rankings.append(ranking_df)

        # Evaluate performance with full data (but imputed)
        model = XGBRegressor(n_estimators=50, max_depth=4, random_state=RANDOM_STATE)
        model.fit(X_train_imputed[:, rfe.support_], y_train.iloc[:, i])

        y_pred = model.predict(X_test_imputed[:, rfe.support_])
        r2 = r2_score(y_test.iloc[:, i], y_pred)

        performance_results[target] = {
            'r2_score': r2,
            'n_features': len(target_features),
            'selected_features': target_features
        }

        print(f"      ✅ {target}: {len(target_features)} features, R² = {r2:.3f}")

        # Clean memory
        gc.collect()

    # Get union of all selected features
    final_selected_features = list(set(all_selected_features))

    # Combine rankings
    combined_ranking = pd.concat(feature_rankings, ignore_index=True)

    print(f"🔚 RFE selected {len(final_selected_features)} unique features")

    return final_selected_features, combined_ranking, performance_results


def random_forest_feature_selection_chunked(X, y, n_features=RF_N_FEATURES):
    """Feature selection using Random Forest with memory optimization"""
    print(f"\n🌲 Performing Random Forest feature selection ({n_features} features)...")

    if len(X) < 50:
        print(f"⚠️  Not enough data for Random Forest ({len(X)} samples)")
        return list(X.columns), pd.DataFrame(), {}

    # Use subset for memory efficiency
    if len(X) > 10000:
        print(f"   📉 Subsampling to 10,000 samples for memory efficiency")
        X_sample = X.sample(n=10000, random_state=RANDOM_STATE)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train Random Forest with reduced parameters
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=50,  # Reduced for memory
            max_depth=4,  # Reduced for memory
            random_state=RANDOM_STATE,
            n_jobs=1  # Single job to reduce memory
        )
    )

    rf.fit(X_train_imputed, y_train)

    # Calculate feature importance (average across all targets)
    importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select top features
    selected_features = importance_df.head(n_features)['feature'].tolist()

    # Evaluate performance with selected features
    performance_results = {}

    # Get indices of selected features
    selected_indices = [list(X.columns).index(feat) for feat in selected_features]

    for i, target in enumerate(TARGETS):
        # Train model with selected features
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=RANDOM_STATE)
        model.fit(X_train_imputed[:, selected_indices], y_train.iloc[:, i])

        # Predict and evaluate
        y_pred = model.predict(X_test_imputed[:, selected_indices])
        r2 = r2_score(y_test.iloc[:, i], y_pred)

        performance_results[target] = {
            'r2_score': r2,
            'n_features': len(selected_features),
            'selected_features': selected_features
        }

        print(f"   🎯 {target}: R² = {r2:.3f}")

    print(f"🔚 Random Forest selected {len(selected_features)} features")

    return selected_features, importance_df, performance_results


def correlation_based_selection_chunked(X, y, n_features=15):
    """Correlation-based selection with chunked processing"""
    print(f"\n📊 Performing correlation-based selection ({n_features} features)...")

    # Process in chunks to save memory
    chunk_size = MAX_FEATURES_PER_CHUNK
    correlation_scores = []

    features = list(X.columns)

    for i in range(0, len(features), chunk_size):
        chunk_features = features[i:i + chunk_size]
        print(f"   🔄 Processing correlation chunk {i // chunk_size + 1}/{(len(features) - 1) // chunk_size + 1}")

        for feature in chunk_features:
            feature_scores = []
            for target in TARGETS:
                # Calculate correlation on complete cases
                valid_data = pd.concat([X[feature], y[target]], axis=1).dropna()
                if len(valid_data) > 10:
                    corr = valid_data[feature].corr(valid_data[target])
                    if not np.isnan(corr):
                        feature_scores.append(abs(corr))

            if feature_scores:
                avg_correlation = np.mean(feature_scores)
                completeness = X[feature].notna().sum() / len(X)
                correlation_scores.append({
                    'feature': feature,
                    'avg_correlation': avg_correlation,
                    'completeness': completeness,
                    'weighted_score': avg_correlation * completeness
                })

        gc.collect()

    # Create correlation dataframe
    correlation_df = pd.DataFrame(correlation_scores)
    if correlation_df.empty:
        print("⚠️  No correlation scores calculated")
        return list(X.columns), pd.DataFrame()

    correlation_df = correlation_df.sort_values('weighted_score', ascending=False)

    # Select top features
    selected_features = correlation_df.head(n_features)['feature'].tolist()

    print(f"🔚 Correlation-based selected {len(selected_features)} features")

    return selected_features, correlation_df


# =============================================================================
# RESULTS EXPORT (Same as before)
# =============================================================================

def export_feature_selection_results(method_name, selected_features, ranking_df=None,
                                     importance_df=None, performance_results=None):
    """Export feature selection results to files"""
    method_dir = OUTPUT_DIR / method_name
    os.makedirs(method_dir, exist_ok=True)

    # Save selected features
    features_df = pd.DataFrame({'selected_features': selected_features})
    features_df.to_csv(method_dir / 'selected_features.csv', index=False)

    # Save ranking/importance
    if ranking_df is not None and not ranking_df.empty:
        ranking_df.to_csv(method_dir / 'feature_ranking.csv', index=False)

    if importance_df is not None and not importance_df.empty:
        importance_df.to_csv(method_dir / 'feature_importance.csv', index=False)

    # Save performance results
    if performance_results:
        performance_data = []
        for target, results in performance_results.items():
            performance_data.append({
                'target': target,
                'r2_score': results['r2_score'],
                'n_features': results['n_features']
            })
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(method_dir / 'performance_metrics.csv', index=False)

    print(f"💾 {method_name} results saved to: {method_dir}/")


def create_comparison_report(all_results):
    """Create a comparison report of all feature selection methods"""
    print("\n📋 Creating comparison report...")

    comparison_data = []

    for method_name, results in all_results.items():
        selected_features = results['selected_features']
        performance = results.get('performance', {})

        # Calculate average R² across targets
        r2_scores = [perf['r2_score'] for perf in performance.values() if 'r2_score' in perf]
        avg_r2 = np.mean(r2_scores) if r2_scores else 0

        comparison_data.append({
            'method': method_name,
            'n_features': len(selected_features),
            'avg_r2_score': avg_r2,
            'selected_features': ', '.join(selected_features[:5]) + '...' if len(selected_features) > 5 else ', '.join(
                selected_features)
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('avg_r2_score', ascending=False)

    # Save comparison
    comparison_df.to_csv(OUTPUT_DIR / 'method_comparison.csv', index=False)

    # Create summary report
    report = f"""
FEATURE SELECTION COMPARISON REPORT
===================================

Total methods evaluated: {len(all_results)}

Performance Ranking:
"""
    for i, row in comparison_df.iterrows():
        report += f"{i + 1}. {row['method']}: {row['n_features']} features, Avg R² = {row['avg_r2_score']:.3f}\n"

    report += f"\nOutput Directory: {OUTPUT_DIR}"

    with open(OUTPUT_DIR / 'feature_selection_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return comparison_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main feature selection execution with memory optimization"""
    print("=" * 60)
    print("MEMORY-EFFICIENT FEATURE SELECTION ANALYSIS")
    print("Using chunked processing to prevent memory issues")
    print("=" * 60)

    try:
        # Step 1: Create processed dataset with chunked processing
        df, candidate_variables = create_processed_dataset_chunked()

        if df is None:
            print("❌ Failed to create processed dataset")
            return

        # Step 2: Prepare data for feature selection
        print("\n📊 Preparing data for feature selection...")

        # Use candidate variables or all available numeric features
        exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study', 'date']
        feature_columns = [col for col in df.columns if
                           col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        X = df[feature_columns].copy()
        y = df[TARGETS].copy()

        # Remove rows where targets are missing
        valid_idx = y.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"✅ Final dataset for feature selection: {X.shape}")
        print(f"💾 Final memory usage: {X.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        if len(X) < 50:
            print("⚠️  Not enough samples for feature selection")
            return

        # Step 3: Perform feature selection methods with memory optimization
        all_results = {}

        # Method 1: RFE Feature Selection
        print("\n" + "=" * 50)
        print("METHOD 1: RECURSIVE FEATURE ELIMINATION (RFE)")
        print("=" * 50)
        rfe_features, rfe_ranking, rfe_performance = rfe_feature_selection_chunked(X, y, RFE_N_FEATURES)
        all_results['rfe'] = {
            'selected_features': rfe_features,
            'ranking': rfe_ranking,
            'performance': rfe_performance
        }
        export_feature_selection_results('rfe', rfe_features, rfe_ranking,
                                         performance_results=rfe_performance)

        # Method 2: Random Forest Feature Selection
        print("\n" + "=" * 50)
        print("METHOD 2: RANDOM FOREST IMPORTANCE")
        print("=" * 50)
        rf_features, rf_importance, rf_performance = random_forest_feature_selection_chunked(X, y, RF_N_FEATURES)
        all_results['random_forest'] = {
            'selected_features': rf_features,
            'importance': rf_importance,
            'performance': rf_performance
        }
        export_feature_selection_results('random_forest', rf_features,
                                         importance_df=rf_importance,
                                         performance_results=rf_performance)

        # Method 3: Correlation-based Selection
        print("\n" + "=" * 50)
        print("METHOD 3: CORRELATION-BASED SELECTION")
        print("=" * 50)
        corr_features, corr_df = correlation_based_selection_chunked(X, y, 15)
        all_results['correlation'] = {
            'selected_features': corr_features,
            'correlation_scores': corr_df
        }
        export_feature_selection_results('correlation', corr_features,
                                         importance_df=corr_df)

        # Method 4: Union of all methods
        print("\n" + "=" * 50)
        print("METHOD 4: UNION OF ALL METHODS")
        print("=" * 50)
        union_features = list(set(rfe_features + rf_features + corr_features))
        all_results['union'] = {
            'selected_features': union_features
        }
        export_feature_selection_results('union', union_features)

        # Create comparison report
        comparison_df = create_comparison_report(all_results)

        print("\n✅ FEATURE SELECTION COMPLETE!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print("\n🎯 NEXT STEPS:")
        print("1. Check the selected features in each method's folder")
        print("2. Use 'selected_features.csv' in your next model")
        print("3. Compare performance using 'method_comparison.csv'")

        return all_results, comparison_df

    except Exception as e:
        print(f"❌ Error in feature selection: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, comparison = main()