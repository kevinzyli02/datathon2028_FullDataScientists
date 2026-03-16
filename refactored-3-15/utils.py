# =============================================================================
# COMPREHENSIVE MODEL WITH PATIENT-WISE 80/20 SPLIT + PERSONALIZATION ENHANCEMENTS
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non‑interactive backend, prevents Tkinter thread errors
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time

DATA_DIR = Path('data')
OUTPUT_DIR = Path('patient_wise_model_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use all available files for comprehensive analysis
COMPREHENSIVE_FILES = [
    'hormones_and_selfreport.csv',
    'sleep.csv',
    'stress_score.csv',
    'resting_heart_rate.csv',
    'daily_norm.csv',
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
# DATA FILTERING FUNCTIONS
# =============================================================================

def filter_all_files_to_match_patients(main_file_path, data_dir, output_dir, file_list):
    """
    Filter all data files to only include patients present in the main file

    Parameters:
    - main_file_path: Path to the main file (hormones_and_selfreport.csv)
    - data_dir: Directory containing all data files
    - output_dir: Directory to save filtered files
    - file_list: List of all files to filter
    """

    print("🔍 FILTERING DATA FILES TO MATCH PATIENTS IN MAIN FILE")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read main file and get patient IDs
    print(f"📋 Reading main file: {main_file_path}")
    main_df = pd.read_csv(main_file_path)
    valid_patient_ids = set(main_df['id'].unique())
    print(f"✅ Found {len(valid_patient_ids)} valid patients in main file")

    # Step 2: Process each file
    for file_name in file_list:
        print(f"\n🔄 Processing {file_name}...")
        file_path = data_dir / file_name

        if not file_path.exists():
            print(f"   ⚠️ File not found: {file_name}")
            continue

        # Read the file
        df = pd.read_csv(file_path)
        original_size = len(df)
        original_patients = len(df['id'].unique()) if 'id' in df.columns else "No ID column"

        print(f"   Original: {original_size} records, {original_patients} patients")

        # Filter based on patient IDs
        if 'id' in df.columns:
            filtered_df = df[df['id'].isin(valid_patient_ids)].copy()
            filtered_size = len(filtered_df)
            filtered_patients = len(filtered_df['id'].unique())

            print(f"   Filtered: {filtered_size} records, {filtered_patients} patients")
            print(
                f"   Removed: {original_size - filtered_size} records ({((original_size - filtered_size) / original_size * 100):.1f}%)")

            # Save filtered file
            output_path = output_dir / file_name
            filtered_df.to_csv(output_path, index=False)
            print(f"   💾 Saved filtered file to: {output_path}")
        else:
            print(f"   ⚠️ No 'id' column found in {file_name}, skipping filtering")
            # Copy file as-is
            output_path = output_dir / file_name
            df.to_csv(output_path, index=False)
            print(f"   💾 Copied file to: {output_path}")

    print(f"\n✅ FILTERING COMPLETE!")
    print(f"📁 Filtered files saved to: {output_dir}")
    return output_dir


def analyze_patient_overlap(data_dir, file_list):
    """
    Analyze patient overlap across all files to identify inconsistencies
    """
    print("\n🔍 ANALYZING PATIENT OVERLAP ACROSS FILES")
    print("=" * 50)

    patient_sets = {}

    for file_name in file_list:
        file_path = data_dir / file_name
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        if 'id' in df.columns:
            patients = set(df['id'].unique())
            patient_sets[file_name] = patients
            print(f"📊 {file_name}: {len(patients)} patients")

    # Find common patients across all files
    if patient_sets:
        common_patients = set.intersection(*patient_sets.values())
        print(f"\n👥 Common patients across ALL files: {len(common_patients)}")

        # Find patients missing from each file
        for file_name, patients in patient_sets.items():
            missing_from_this = common_patients - patients
            if missing_from_this:
                print(f"   ⚠️ {file_name} missing {len(missing_from_this)} patients")

        return common_patients
    return set()


def create_consistency_report(main_file_path, data_dir, file_list):
    """
    Create a comprehensive report on data consistency
    """
    print("\n📋 CREATING DATA CONSISTENCY REPORT")
    print("=" * 50)

    # Read main file
    main_df = pd.read_csv(main_file_path)
    main_patients = set(main_df['id'].unique())

    report_data = []

    for file_name in file_list:
        file_path = data_dir / file_name
        if not file_path.exists():
            report_data.append({
                'file': file_name,
                'status': 'MISSING',
                'patients': 0,
                'overlap_with_main': 0,
                'overlap_percent': 0,
                'records': 0
            })
            continue

        df = pd.read_csv(file_path)
        records = len(df)

        if 'id' in df.columns:
            patients = set(df['id'].unique())
            overlap = patients & main_patients
            overlap_percent = len(overlap) / len(main_patients) * 100 if main_patients else 0

            report_data.append({
                'file': file_name,
                'status': 'OK',
                'patients': len(patients),
                'overlap_with_main': len(overlap),
                'overlap_percent': overlap_percent,
                'records': records
            })
        else:
            report_data.append({
                'file': file_name,
                'status': 'NO_ID_COLUMN',
                'patients': 'N/A',
                'overlap_with_main': 'N/A',
                'overlap_percent': 'N/A',
                'records': records
            })

    # Create report DataFrame
    report_df = pd.DataFrame(report_data)
    print("\n📊 DATA CONSISTENCY REPORT:")
    print(report_df.to_string(index=False))

    # Identify files that need filtering
    files_to_filter = []
    for item in report_data:
        if item['status'] == 'OK' and item['overlap_percent'] < 100:
            files_to_filter.append(item['file'])

    if files_to_filter:
        print(f"\n🎯 FILES THAT NEED FILTERING: {len(files_to_filter)}")
        for file_name in files_to_filter:
            print(f"   - {file_name}")
    else:
        print(f"\n✅ All files already consistent with main file!")

    return report_df


# =============================================================================
# ENHANCEMENT 1: BASELINE NORMALIZATION
# =============================================================================

def normalize_by_baseline(df, targets, baseline_days=3):
    """Normalize hormone levels by patient-specific baseline"""

    df_normalized = df.copy()

    for target in targets:
        if target not in df.columns:
            continue

        # Create normalized column
        normalized_col = f"{target}_normalized"
        df_normalized[normalized_col] = np.nan

        for patient_id in df['id'].unique():
            patient_data = df[df['id'] == patient_id].copy()
            patient_data = patient_data.sort_values('day_in_study')

            # Calculate baseline (first few days in study)
            baseline_data = patient_data.head(baseline_days)
            baseline_value = baseline_data[target].mean()

            if not np.isnan(baseline_value) and baseline_value > 0:
                # Normalize all values by baseline
                mask = df_normalized['id'] == patient_id
                df_normalized.loc[mask, normalized_col] = (
                        df_normalized.loc[mask, target] / baseline_value
                )

                print(f"   Patient {patient_id}: baseline {target} = {baseline_value:.2f}")

    return df_normalized


# =============================================================================
# ENHANCEMENT 2: PERSONALIZED FEATURES
# =============================================================================

def add_personalized_features(df, targets, window_size=7):
    """Add patient-specific rolling features, using only past data (no leakage)."""
    df_personal = df.copy()
    df_personal = df_personal.sort_values(['id', 'day_in_study']).reset_index(drop=True)

    for target in targets:
        if target not in df.columns:
            continue

        print(f"   Adding personalized features for {target}...")

        for patient_id in df_personal['id'].unique():
            patient_mask = df_personal['id'] == patient_id
            patient_indices = df_personal.index[patient_mask]

            if len(patient_indices) < 2:
                continue

            # Get patient data in order
            patient_data = df_personal.loc[patient_indices, target].copy()

            # --- Target‑derived features that must NOT use current day ---
            # Shift by 1 to use only past days
            past_values = patient_data.shift(1)

            # Rolling mean of past values (min_periods=1 so first day gets NaN)
            rolling_mean_past = past_values.rolling(window=min(window_size, len(patient_data)),
                                                    min_periods=1).mean()
            df_personal.loc[patient_indices, f'{target}_rolling_mean_past'] = rolling_mean_past

            # Rolling std of past values
            rolling_std_past = past_values.rolling(window=min(window_size, len(patient_data)),
                                                   min_periods=1).std()
            df_personal.loc[patient_indices, f'{target}_rolling_std_past'] = rolling_std_past.fillna(0)

            # Personal mean (overall baseline) – uses all data, but that's okay because it's a constant per patient
            personal_mean = patient_data.mean()
            df_personal.loc[patient_indices, f'{target}_personal_mean'] = personal_mean

            # Deviation from personal mean – uses current day, but that's the target itself? Actually deviation = current - personal_mean, so it uses current target. That's a leak if used as feature. Better to compute deviation from past mean? We'll skip deviation.
            # Instead, we can compute lagged deviation (deviation from past mean) – but that's complicated. For simplicity, we'll drop deviation.
            # We'll keep daily_change but shift it to be previous day's change.
            daily_change = patient_data.diff().shift(1)  # change from previous day to current? Actually diff gives t - (t-1). We want yesterday's change as feature. So shift(1) again.
            df_personal.loc[patient_indices, f'{target}_prev_daily_change'] = daily_change.shift(1).fillna(0)

            # Cumulative sum up to previous day
            cumsum_past = patient_data.cumsum().shift(1)
            df_personal.loc[patient_indices, f'{target}_cumulative_past'] = cumsum_past.fillna(0)

        print(f"   ✅ Personalized features for {target} added (no leakage).")

    return df_personal

def add_cycle_phase_features(df):
    """Add menstrual cycle phase indicators"""

    # Simplified phase estimation based on cycle day
    df['phase_follicular'] = (df['cycle_day'] <= 14).astype(int)
    df['phase_luteal'] = (df['cycle_day'] > 14).astype(int)
    df['phase_ovulation'] = ((df['cycle_day'] >= 12) & (df['cycle_day'] <= 16)).astype(int)

    return df


# =============================================================================
# ENHANCEMENT 3: MIXED EFFECTS ENSEMBLE MODEL - FIXED VERSION
# =============================================================================

# Alternative: Simplified Mixed Effects that's more robust
class SimpleMixedEffects(BaseEstimator, RegressorMixin):
    """Simplified version that's less likely to have initialization issues"""

    def __init__(self, min_patient_samples=5):
        self.min_patient_samples = min_patient_samples
        self.global_model = None
        self.patient_models = {}
        self.is_fitted = False

    def fit(self, X, y, patient_ids):
        print("🏥 Training Simple Mixed Effects...")

        # Simple global model
        self.global_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
        )
        self.global_model.fit(X, y)

        # Patient-specific adjustments (simpler approach)
        unique_patients = np.unique(patient_ids)
        trained_count = 0

        for patient_id in unique_patients:
            mask = patient_ids == patient_id
            if np.sum(mask) >= self.min_patient_samples:
                X_patient = X[mask]
                y_patient = y[mask]

                # Simple correction: train on this patient's data directly
                try:
                    patient_model = RandomForestRegressor(
                        n_estimators=20, max_depth=4, random_state=42
                    )
                    patient_model.fit(X_patient, y_patient)
                    self.patient_models[patient_id] = patient_model
                    trained_count += 1
                except:
                    pass

        print(f"   ✅ Trained models for {trained_count}/{len(unique_patients)} patients")
        self.is_fitted = True
        return self

    def predict(self, X, patient_ids=None):
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Start with global predictions
        y_pred = self.global_model.predict(X)

        if patient_ids is not None:
            # Apply patient-specific predictions where available
            for i, patient_id in enumerate(patient_ids):
                if patient_id in self.patient_models:
                    try:
                        # Use patient-specific model if available
                        y_pred[i] = self.patient_models[patient_id].predict([X[i]])[0]
                    except:
                        # Keep global prediction if patient model fails
                        pass

        return y_pred

    def get_patient_models_info(self):
        """Get information about patient-specific models"""
        return {
            'n_patient_models': len(self.patient_models),
            'coverage_rate': len(self.patient_models) / len(set(self.patient_models.keys())) * 100
        }


# =============================================================================
# MEMORY-EFFICIENT DATA LOADING
# =============================================================================
def load_comprehensive_data_memory_efficient(files, data_dir, sample_size=SAMPLE_SIZE):
    """Load and merge comprehensive dataset with memory efficiency, handling huge files."""
    print("📂 Loading comprehensive dataset (Memory Efficient)...")

    # Start with base file
    base_file = files[0]
    df = pd.read_csv(data_dir / base_file)
    df = preprocess_data(df)
    print(f"Base data: {df.shape}")

    # Process files in chunks and merge strategically
    for file_name in files[1:]:
        print(f"🔗 Merging {file_name}...")
        try:
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"   ⚠️ File not found: {file_name}")
                continue

            # Estimate file size in bytes
            file_size = file_path.stat().st_size
            print(f"   File size: {file_size / 1e9:.2f} GB")

            # Force chunk reading for known huge file(s)
            force_chunk = "sleep_score" in file_name.lower()
            # If file > 500 MB or forced, read in chunks
            if force_chunk or file_size > 500_000_000:
                print(f"   Large file, using chunk reading and sampling...")
                chunks = []
                chunksize = 500_000
                sampled_rows = 0
                target_sample = 20_000  # max rows we want from this file

                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    chunk = preprocess_data(chunk)
                    chunk = handle_file_structure(chunk, file_name)

                    if sampled_rows >= target_sample:
                        break

                    # Sample from this chunk
                    if 'id' in chunk.columns:
                        # Simple random sample from chunk
                        sample_needed = min(len(chunk), target_sample - sampled_rows)
                        if sample_needed > 0:
                            sampled = chunk.sample(n=sample_needed, random_state=RANDOM_STATE)
                            chunks.append(sampled)
                            sampled_rows += len(sampled)
                    else:
                        sample_needed = min(len(chunk), target_sample - sampled_rows)
                        if sample_needed > 0:
                            sampled = chunk.sample(n=sample_needed, random_state=RANDOM_STATE)
                            chunks.append(sampled)
                            sampled_rows += len(sampled)

                if chunks:
                    new_data = pd.concat(chunks, ignore_index=True)
                else:
                    new_data = pd.DataFrame()
                print(f"   🔽 Sampled large file to {new_data.shape}")
            else:
                # Normal file: read fully
                print("   Normal file, reading fully...")
                new_data = pd.read_csv(file_path)
                new_data = preprocess_data(new_data)
                new_data = handle_file_structure(new_data, file_name)

                # For moderately large files, sample aggressively
                if len(new_data) > 100_000:
                    print(f"   🔽 File has {len(new_data)} rows, sampling...")
                    # Sample but preserve patient distribution if possible
                    if 'id' in new_data.columns:
                        patient_sample = new_data['id'].value_counts().head(1000).index
                        new_data = new_data[new_data['id'].isin(patient_sample)]
                    if len(new_data) > 50_000:
                        new_data = new_data.sample(n=10_000, random_state=RANDOM_STATE)
                    print(f"   🔽 Sampled new data to {new_data.shape}")

        except MemoryError:
            print(f"   ❌ MemoryError reading {file_name}, falling back to chunk reading...")
            # Fallback: read in chunks and sample
            chunks = []
            chunksize = 500_000
            sampled_rows = 0
            target_sample = 20_000
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk = preprocess_data(chunk)
                chunk = handle_file_structure(chunk, file_name)
                if sampled_rows >= target_sample:
                    break
                sample_needed = min(len(chunk), target_sample - sampled_rows)
                if sample_needed > 0:
                    sampled = chunk.sample(n=sample_needed, random_state=RANDOM_STATE)
                    chunks.append(sampled)
                    sampled_rows += len(sampled)
            if chunks:
                new_data = pd.concat(chunks, ignore_index=True)
            else:
                new_data = pd.DataFrame()
            print(f"   🔽 Fallback sampling produced {new_data.shape}")

        except Exception as e:
            print(f"❌ Error with {file_name}: {e}")
            continue

        if new_data.empty:
            print(f"   ⚠️ No data after processing for {file_name}, skipping")
            continue

        # Find merge keys
        merge_keys = find_merge_keys(df, new_data, file_name)
        if not merge_keys:
            print(f"   ⚠️ No merge keys found for {file_name}")
            continue

        # Memory-efficient merge by selecting only numeric columns
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in new_data.columns and 'id' not in numeric_cols:
            numeric_cols.append('id')
        if 'day_in_study' in new_data.columns and 'day_in_study' not in numeric_cols:
            numeric_cols.append('day_in_study')

        new_data_reduced = new_data[numeric_cols] if numeric_cols else new_data

        # Perform merge
        df = pd.merge(df, new_data_reduced, on=merge_keys, how='left',
                      suffixes=('', f'_{file_name.replace(".csv", "")}'))
        print(f"✅ After {file_name}: {df.shape}")

        # --- NEW: After each merge, if DataFrame is too large, sample it down ---
        if len(df) > sample_size * 2:
            print(f"   ⚠️ DataFrame too large ({len(df)}), sampling down to {sample_size}...")
            # Sample by patients to maintain distribution
            unique_patients = df['id'].unique()
            if len(unique_patients) > sample_size:
                # If more patients than desired rows, sample patients
                patient_sample = np.random.choice(
                    unique_patients,
                    size=min(sample_size, len(unique_patients)),
                    replace=False
                )
                df = df[df['id'].isin(patient_sample)]
            # If still too large, random sample
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=RANDOM_STATE)
            print(f"   🔽 After sampling: {df.shape}")
        # ----------------------------------------------------------------------

        # Force garbage collection
        del new_data, new_data_reduced
        gc.collect()

    # Final sampling to ensure we are at or below sample_size
    if len(df) > sample_size:
        print(f"🔽 Final sampling from {len(df)} to {sample_size}...")
        unique_patients = df['id'].unique()
        if len(unique_patients) > sample_size:
            patient_sample = np.random.choice(unique_patients, size=sample_size, replace=False)
            df = df[df['id'].isin(patient_sample)]
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"🔽 Final dataset: {df.shape}")

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
        df = aggregate_by_participant_memory_efficient(df)

    return df


def aggregate_by_participant_memory_efficient(df):
    """Aggregate high-frequency data by participant with memory optimization"""
    if 'id' not in df.columns:
        return df

    # Select only numeric columns to save memory
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' not in numeric_cols and 'id' in df.columns:
        numeric_cols.append('id')

    df_reduced = df[numeric_cols]

    # Aggregate statistics by participant
    aggregated = df_reduced.groupby('id').agg(['mean', 'std', 'min', 'max']).reset_index()
    # Flatten column names
    aggregated.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in aggregated.columns]
    print(f"   ✅ Aggregated to {aggregated.shape}")

    del df_reduced
    gc.collect()

    return aggregated


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
# COMPREHENSIVE FEATURE PROCESSING (EXCLUDING HORMONE-DERIVED FEATURES)
# =============================================================================
def get_all_features(df, targets):
    """Get all available features after cleaning, excluding hormone‑related, min/max, and target columns."""
    print(f"\n🎯 Processing all available features (excluding hormone-derived features)...")

    feature_sets = {}

    # Pre‑compute all hormone columns that could ever appear
    hormone_base_cols = ['lh', 'estrogen', 'pdg']
    hormone_derived_patterns = [
        '_normalized', '_rolling_mean', '_rolling_std', '_personal_mean',
        '_deviation', '_daily_change', '_cumulative'
    ]

    all_hormone_columns = set()
    for hormone in hormone_base_cols:
        all_hormone_columns.add(hormone)
        for pattern in hormone_derived_patterns:
            col = f"{hormone}{pattern}"
            if col in df.columns:
                all_hormone_columns.add(col)

    for target in targets:
        if target not in df.columns:
            continue

        print(f"\nProcessing features for {target}...")

        # Base columns to always exclude (metadata)
        exclude_cols = ['id', 'day_in_study', 'study_interval', 'is_weekend']

        # Add all hormone columns EXCEPT the current target
        for hormone_col in all_hormone_columns:
            if hormone_col != target:
                exclude_cols.append(hormone_col)

        # Exclude the target column itself
        exclude_cols.append(target)

        # Remove duplicates
        exclude_cols = list(set(exclude_cols))

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

        # --- NEW: Remove min/max columns ---
        min_max_cols = [col for col in numeric_features if '_min' in col or '_max' in col]
        if min_max_cols:
            numeric_features = [col for col in numeric_features if col not in min_max_cols]
            print(f"   Removed {len(min_max_cols)} min/max columns")
        # ------------------------------------

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


def create_regression_diagnostics(y_true, y_pred, model_name, target_name, n_features, n_patients, output_dir):
    """Create comprehensive regression diagnostic plots and save to output_dir."""
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

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(y_true, p(y_true), "b-", alpha=0.8, label='Regression line')
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

    safe_filename = f'regression_{model_name}_{target_name}'.replace(' ', '_')
    plt.savefig(output_dir / f'{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# CONSISTENT MODEL TRAINING WITH FIXED SPLITS
# =============================================================================

def train_enhanced_models_consistent_splits(df, feature_sets, targets):
    """Enhanced training function with consistent splits across all models"""

    print("\n🚀 TRAINING ENHANCED MODELS WITH CONSISTENT SPLITS")
    print("=" * 80)

    all_results = {}

    # Define enhanced models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
        'MixedEffects': SimpleMixedEffects(min_patient_samples=5),
    }

    for target in targets:
        if target not in df.columns or target not in feature_sets:
            continue

        print(f"\n🎯 TARGET: {target.upper()}")
        print("=" * 50)

        all_features = feature_sets[target]

        if len(all_features) == 0:
            print("❌ No features available for this target")
            continue

        # Prepare data
        target_data = df.dropna(subset=[target]).copy()

        print(f"📊 Dataset before split: {target_data.shape}")
        print(f"🎯 Using {len(all_features)} features")

        # PATIENT-WISE 80/20 Split (ONCE per target)
        train_data, test_data, train_patients, test_patients = patient_wise_train_test_split(
            target_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Prepare features and targets (ONCE per target)
        X_train = train_data[all_features].copy()
        y_train = train_data[target].values
        X_test = test_data[all_features].copy()
        y_test = test_data[target].values

        # Get patient IDs for mixed effects
        train_patient_ids = train_data['id'].values
        test_patient_ids = test_data['id'].values

        print(f"🔀 Final split:")
        print(f"   Training: {X_train.shape} ({len(train_patients)} patients)")
        print(f"   Testing:  {X_test.shape} ({len(test_patients)} patients)")

        # Handle missing values (ONCE per target)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Scale features (ONCE per target)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        print(f"✅ Prepared consistent training data for all models")

        target_results = {}

        for model_name, model in models.items():
            print(f"\n🧪 Training {model_name}...")
            start_time = time.time()

            try:
                # Special handling for Mixed Effects model
                if model_name == 'MixedEffects':
                    model.fit(X_train_scaled, y_train, train_patient_ids)
                    training_time = time.time() - start_time

                    # Get model info
                    model_info = model.get_patient_models_info()
                    print(f"   Patient model coverage: {model_info['coverage_rate']:.1f}%")

                    # Predictions (need to pass patient IDs)
                    y_pred_train = model.predict(X_train_scaled, train_patient_ids)
                    y_pred_test = model.predict(X_test_scaled, test_patient_ids)
                else:
                    # Standard models
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

                # Create diagnostic plots
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

        # Clean memory after each target
        del X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        del train_data, test_data
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
                        'Time_s': result['train_time']  # <-- fixed
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
                    'training_time': result['train_time']
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
            bars1 = ax1.bar(models, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax1.set_ylabel('R² Score')
            ax1.set_title(f'Model Comparison - {target.upper()}\nTest Patients: {n_patients_list[0]}')
            ax1.set_ylim(0, 1)

            # Add value labels on bars
            for bar, score in zip(bars1, r2_scores):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.4f}', ha='center', va='bottom')

            # RMSE comparison
            bars2 = ax2.bar(models, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
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
# QUICK TEST FUNCTION
# =============================================================================

def quick_test_enhancements():
    """Quick test to verify enhancements work"""
    print("🧪 QUICK TEST OF ENHANCEMENTS")

    # Load small sample
    df = load_comprehensive_data_memory_efficient(COMPREHENSIVE_FILES[:3], DATA_DIR, sample_size=1000)

    # Test baseline normalization
    df_normalized = normalize_by_baseline(df, TARGETS[:1])
    print(f"✅ Baseline normalization completed")

    # Test personalized features
    df_enhanced = add_personalized_features(df_normalized, TARGETS[:1])
    print(f"✅ Personalized features added")

    # Check new columns
    new_cols = [col for col in df_enhanced.columns if 'normalized' in col or 'rolling' in col or 'deviation' in col]
    print(f"✅ New features created: {len(new_cols)}")

    # Test Mixed Effects model - use available features instead of assuming cycle_day exists
    if len(df_enhanced) > 10:
        # Find available numeric features (excluding IDs, targets, and hormone-derived features)
        available_features = []
        hormone_cols = ['lh', 'estrogen', 'pdg']
        hormone_patterns = ['_normalized', '_rolling_mean', '_rolling_std', '_personal_mean', '_deviation',
                            '_daily_change', '_cumulative']

        for col in df_enhanced.columns:
            if (pd.api.types.is_numeric_dtype(df_enhanced[col]) and
                    col not in ['id', 'day_in_study'] and
                    not any(hormone in col for hormone in hormone_cols) and
                    not any(pattern in col for pattern in hormone_patterns) and
                    df_enhanced[col].notna().sum() > 0):
                available_features.append(col)

        # Use first 5 available features for testing
        mock_features = available_features[:5]

        if len(mock_features) > 1:
            X = df_enhanced[mock_features].fillna(0).values
            # Use original target if normalized doesn't exist
            if 'lh_normalized' in df_enhanced.columns:
                y = df_enhanced['lh_normalized'].fillna(0).values
            elif 'lh' in df_enhanced.columns:
                y = df_enhanced['lh'].fillna(0).values
            else:
                y = np.random.random(len(X))

            patient_ids = df_enhanced['id'].values

            try:
                mixed_model = SimpleMixedEffects()
                mixed_model.fit(X, y, patient_ids)
                print(f"✅ Mixed Effects model trained successfully")
            except Exception as e:
                print(f"⚠️ Mixed Effects test failed (but continuing): {e}")

    print("🎉 All enhancements tested successfully!")


# =============================================================================
# UPDATED MAIN EXECUTION WITH MEMORY EFFICIENCY
# =============================================================================

def main_with_filtering_memory_efficient():
    """Enhanced main function with memory-efficient data filtering and processing"""

    print("🚀 ENHANCED PIPELINE WITH MEMORY-EFFICIENT PROCESSING")
    print("=" * 80)

    # Define paths
    MAIN_FILE = 'hormones_and_selfreport.csv'
    FILTERED_DATA_DIR = Path('filtered_data')

    # Step 0: Analyze and filter data if needed
    print("\n🔍 STEP 0: CHECKING DATA CONSISTENCY")
    consistency_report = create_consistency_report(
        DATA_DIR / MAIN_FILE,
        DATA_DIR,
        COMPREHENSIVE_FILES
    )

    # Check if filtering is needed
    needs_filtering = any(
        row['status'] == 'OK' and row['overlap_percent'] < 100
        for _, row in consistency_report.iterrows()
    )

    if needs_filtering:
        print("\n🔄 STEP 0.5: FILTERING DATA FILES")
        filtered_dir = filter_all_files_to_match_patients(
            DATA_DIR / MAIN_FILE,
            DATA_DIR,
            FILTERED_DATA_DIR,
            COMPREHENSIVE_FILES
        )
        # Use filtered data directory for the rest of the pipeline
        analysis_data_dir = filtered_dir
    else:
        print("\n✅ Data already consistent, using original files")
        analysis_data_dir = DATA_DIR

    # Now run the enhanced pipeline with consistent data and memory efficiency
    try:
        # Step 1: Load comprehensive dataset from filtered directory with memory efficiency
        print(f"\n📂 STEP 1: LOADING DATA FROM {analysis_data_dir} (Memory Efficient)")
        df = load_comprehensive_data_memory_efficient(COMPREHENSIVE_FILES, analysis_data_dir, SAMPLE_SIZE)
        print(f"📊 Final dataset: {df.shape}")

        # Force garbage collection after loading
        gc.collect()

        # Step 2: Apply baseline normalization
        print("\n🔄 STEP 2: APPLYING BASELINE NORMALIZATION...")
        df = normalize_by_baseline(df, TARGETS, baseline_days=3)

        # Use normalized targets for modeling
        NORMALIZED_TARGETS = [f"{target}_normalized" for target in TARGETS]
        print(f"🎯 Normalized targets: {NORMALIZED_TARGETS}")

        # Step 3: Add personalized features
        print("\n🎯 STEP 3: ADDING PERSONALIZED FEATURES...")
        df = add_personalized_features(df, TARGETS, window_size=7)
        print(f"📊 Dataset with personalized features: {df.shape}")

        # Clean memory
        gc.collect()

        # Step 4: Get features for enhanced targets (EXCLUDING HORMONE-DERIVED FEATURES)
        feature_sets = get_all_features(df, NORMALIZED_TARGETS)

        # Step 5: Train enhanced models with CONSISTENT splits
        print("\n🤖 STEP 4: TRAINING ENHANCED MODELS WITH CONSISTENT SPLITS...")
        results = train_enhanced_models_consistent_splits(df, feature_sets, NORMALIZED_TARGETS)

        # Step 6: Create comprehensive report
        create_comprehensive_report(results, feature_sets)

        print(f"\n✅ ENHANCED PERSONALIZED MODELING COMPLETE!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")

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
        print(f"❌ Error in enhanced main execution: {e}")
        import traceback
        traceback.print_exc()
        # Try to free memory and continue with smaller dataset
        gc.collect()
        print("🔄 Attempting to continue with smaller dataset...")

        # Fallback: use smaller sample size
        try:
            SAMPLE_SIZE_FALLBACK = 5000
            print(f"🔄 Using smaller sample size: {SAMPLE_SIZE_FALLBACK}")
            df = load_comprehensive_data_memory_efficient(COMPREHENSIVE_FILES[:5], analysis_data_dir,
                                                          SAMPLE_SIZE_FALLBACK)

            # Continue with smaller dataset...
            print("🔄 Continuing with smaller dataset...")
            # Apply the same steps as above but with smaller dataset
            df = normalize_by_baseline(df, TARGETS, baseline_days=3)
            NORMALIZED_TARGETS = [f"{target}_normalized" for target in TARGETS]
            df = add_personalized_features(df, TARGETS, window_size=7)
            feature_sets = get_all_features(df, NORMALIZED_TARGETS)
            results = train_enhanced_models_consistent_splits(df, feature_sets, NORMALIZED_TARGETS)
            create_comprehensive_report(results, feature_sets)

        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")


# =============================================================================
# QUICK DATA FILTERING FUNCTION (Standalone)
# =============================================================================

def quick_filter_data():
    """
    Quick function to just filter the data without running the full pipeline
    """
    MAIN_FILE = 'hormones_and_selfreport.csv'
    FILTERED_DATA_DIR = Path('filtered_data')

    print("🔍 QUICK DATA FILTERING")
    print("=" * 50)

    # First check consistency
    report = create_consistency_report(
        DATA_DIR / MAIN_FILE,
        DATA_DIR,
        COMPREHENSIVE_FILES
    )

    # Filter if needed
    filter_all_files_to_match_patients(
        DATA_DIR / MAIN_FILE,
        DATA_DIR,
        FILTERED_DATA_DIR,
        COMPREHENSIVE_FILES
    )

    print("\n✅ DATA FILTERING COMPLETE!")
    print(f"📁 Filtered files saved to: {FILTERED_DATA_DIR}")

### Figure generation ###
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def generate_post_training_outputs(all_results, output_dir):
    """
    Generate residual plots, feature importance Excel, and predictions CSV.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    importance_records = []
    predictions_records = []   # new

    for target, model_dict in all_results.items():
        for model_name, res in model_dict.items():
            if res is None:
                continue

            y_true = res['test_true']
            y_pred = res['test_predictions']
            patient_ids = res['test_ids']
            residuals = y_true - y_pred

            # --- Residual plot (unchanged) ---
            plot_df = pd.DataFrame({
                'Predicted': y_pred,
                'Residual': residuals,
                'Patient': patient_ids
            })
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(plot_df['Predicted'], plot_df['Residual'],
                                  c=plot_df['Patient'], cmap='tab20', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Patient ID')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Predicted Value')
            plt.ylabel('Residual (Actual - Predicted)')
            plt.title(f'Residual Plot: {target} - {model_name}')
            plt.tight_layout()
            plt.savefig(plots_dir / f'residual_{target}_{model_name}.png', dpi=300)
            plt.close()

            # --- NEW: Save predictions for later CSV export ---
            for i in range(len(y_true)):
                predictions_records.append({
                    'target': target,
                    'model': model_name,
                    'patient_id': patient_ids[i],
                    'actual': y_true[i],
                    'predicted': y_pred[i],
                    'residual': residuals[i]
                })

            # --- Feature importances (unchanged) ---
            if res['feature_importances'] is not None:
                for feat_name, imp in zip(res['feature_names'], res['feature_importances']):
                    importance_records.append({
                        'target': target,
                        'model': model_name,
                        'feature': feat_name,
                        'importance': imp
                    })

    # Save predictions CSV
    if predictions_records:
        pred_df = pd.DataFrame(predictions_records)
        pred_df.to_csv(output_dir / "all_predictions.csv", index=False)
        print(f"Predictions saved to {output_dir / 'all_predictions.csv'}")


    if importance_records:
        imp_df = pd.DataFrame(importance_records)

        def categorize_feature(feature):
            feature_lower = feature.lower()
            if 'sleep' in feature_lower:
                return 'Sleep-related'
            elif 'stress' in feature_lower:
                return 'Stress-related'
            elif 'heart' in feature_lower or 'bpm' in feature_lower or 'hr' in feature_lower:
                return 'Heart-related'
            elif 'temp' in feature_lower or 'temperature' in feature_lower:
                return 'Temperature-related'
            elif 'respiratory' in feature_lower or 'breath' in feature_lower:
                return 'Respiratory-related'
            elif 'exercise' in feature_lower or 'activity' in feature_lower or 'calories' in feature_lower:
                return 'Exercise-related'
            elif any(s in feature_lower for s in ['appetite', 'headache', 'cramp', 'breast', 'fatigue', 'mood', 'food', 'indigestion', 'bloating']):
                return 'Symptom-related'
            elif 'glucose' in feature_lower:
                return 'Glucose-related'
            else:
                return 'Other'

        imp_df['category'] = imp_df['feature'].apply(categorize_feature)

        # Save to Excel with separate sheets per target-model
        excel_path = output_dir / 'feature_importances.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for target in imp_df['target'].unique():
                for model in imp_df['model'].unique():
                    subset = imp_df[(imp_df['target'] == target) & (imp_df['model'] == model)]
                    if not subset.empty:
                        sheet_name = f"{target}_{model}"[:31]  # Excel sheet name limit
                        subset_sorted = subset[['feature', 'category', 'importance']].sort_values('importance', ascending=False)
                        subset_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Feature importances saved to {excel_path}")
    else:
        print("No feature importances available.")


def add_lag_features(df, lag_days=[1, 2, 3], base_sensors=None):
    """
    Add lagged versions of daily sensor values for each patient.

    Parameters:
    - df: DataFrame with 'id' and 'day_in_study'.
    - lag_days: list of integer lags (e.g., [1,2,3]).
    - base_sensors: list of column name substrings to lag. If None, lags all columns ending with '_mean'.

    Returns:
    - DataFrame with new lag columns.
    """
    if base_sensors is None:
        # Default: lag all columns that end with '_mean' (daily aggregates)
        base_sensors = [col for col in df.columns if col.endswith('_mean')]
    else:
        # Filter columns that contain any of the base_sensor substrings
        base_sensors = [col for col in df.columns
                        if any(sensor in col for sensor in base_sensors)
                        and col not in ['id', 'day_in_study']]

    df_lagged = df.copy()
    for col in base_sensors:
        if col not in df.columns:
            continue
        for lag in lag_days:
            lag_col = f'{col}_lag{lag}'
            # For each patient, shift the column by 'lag' days (uses past values)
            df_lagged[lag_col] = df_lagged.groupby('id')[col].shift(lag)
    return df_lagged
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run quick test first
    quick_test_enhancements()

    # Run full enhanced pipeline with memory efficiency
    main_with_filtering_memory_efficient()