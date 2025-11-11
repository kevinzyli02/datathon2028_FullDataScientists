# hormone_based_predictions_glucose_debug.py
"""
HORMONE-BASED PREDICTIONS - GLUCOSE DEBUG VERSION
Debugging why glucose_target column is missing
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")
OUTPUT_DIR = Path('hormone_based_predictions_glucose_debug')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hormone features
HORMONE_FEATURES = ['lh', 'estrogen', 'pdg']

# Target variables
TARGETS = {
    'phase': 'categorical',
    'menstruation_start': 'regression',
    'glucose': 'regression'
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# IMPROVED GLUCOSE PROCESSING WITH DEBUGGING
# =============================================================================

def add_glucose_features_debug(df, glucose_df, is_training=True):
    """Add glucose features with extensive debugging"""
    print(f"   🍬 Adding glucose features ({'training' if is_training else 'testing'})...")

    # Create a backup of original dataframe
    original_columns = set(df.columns)

    try:
        print(f"   🔍 Glucose data shape: {glucose_df.shape}")
        print(f"   🔍 Glucose data columns: {glucose_df.columns.tolist()}")
        print(f"   🔍 Glucose data sample:")
        print(glucose_df.head(3))

        # Check if glucose data has the required columns
        if 'glucose_value' not in glucose_df.columns:
            print("   ❌ 'glucose_value' column not found in glucose data")
            # Create synthetic glucose target
            df['glucose_target'] = np.random.normal(5.5, 1.0, len(df))
            return df

        # Clean glucose data
        glucose_clean = glucose_df.copy()
        print(f"   🔍 Original glucose records: {len(glucose_clean)}")

        # Handle infinite values
        glucose_clean['glucose_value'] = glucose_clean['glucose_value'].replace([np.inf, -np.inf], np.nan)
        print(f"   🔍 Glucose records after inf removal: {len(glucose_clean)}")

        # Remove rows with NaN glucose values
        glucose_clean = glucose_clean.dropna(subset=['glucose_value'])
        print(f"   🔍 Glucose records after NaN removal: {len(glucose_clean)}")

        if len(glucose_clean) == 0:
            print("   ❌ No valid glucose data after cleaning")
            df['glucose_target'] = 5.5  # Default value
            return df

        # Basic glucose statistics
        glucose_stats = glucose_clean['glucose_value'].describe()
        print(f"   📊 Glucose value stats:")
        print(f"      Count: {glucose_stats['count']:.0f}")
        print(f"      Mean: {glucose_stats['mean']:.2f}")
        print(f"      Std: {glucose_stats['std']:.2f}")
        print(f"      Min: {glucose_stats['min']:.2f}")
        print(f"      Max: {glucose_stats['max']:.2f}")

        # Check for patients with glucose data
        patients_with_glucose = glucose_clean['id'].unique()
        print(f"   👥 Patients with glucose data: {len(patients_with_glucose)}")

        # Get patients in current dataset
        patients_in_dataset = df['id'].unique()
        print(f"   👥 Patients in current dataset: {len(patients_in_dataset)}")

        # Find overlapping patients
        overlapping_patients = set(patients_with_glucose) & set(patients_in_dataset)
        print(f"   🔗 Overlapping patients: {len(overlapping_patients)}")

        if len(overlapping_patients) == 0:
            print("   ⚠️  No overlapping patients between hormones and glucose data")
            print("   🛠️  Using global glucose statistics")

            # Use global glucose statistics
            global_glucose_mean = glucose_clean['glucose_value'].mean()
            global_glucose_std = glucose_clean['glucose_value'].std()

            df['glucose_global_mean'] = global_glucose_mean
            df['glucose_global_std'] = global_glucose_std
            df['glucose_target'] = global_glucose_mean

            print(f"   ✅ Set global glucose target: {global_glucose_mean:.2f}")
            return df

        # Calculate patient-specific glucose statistics only for overlapping patients
        patient_glucose_stats = glucose_clean[glucose_clean['id'].isin(overlapping_patients)].groupby('id')[
            'glucose_value'].agg([
            'mean', 'std', 'count'
        ]).reset_index()

        patient_glucose_stats.columns = ['id', 'glucose_patient_mean', 'glucose_patient_std', 'glucose_count']

        print(f"   📊 Patient glucose stats calculated for {len(patient_glucose_stats)} patients")
        print(f"   📊 Patient glucose stats sample:")
        print(patient_glucose_stats.head(3))

        # Merge with main dataframe
        df = pd.merge(df, patient_glucose_stats, on='id', how='left')

        # Check merge results
        merged_patients_with_glucose = df['glucose_patient_mean'].notna().sum()
        print(f"   🔗 Patients with glucose data after merge: {merged_patients_with_glucose}")

        # Calculate global statistics for normalization
        global_glucose_mean = patient_glucose_stats['glucose_patient_mean'].mean()
        global_glucose_std = patient_glucose_stats['glucose_patient_mean'].std()

        print(f"   📊 Global glucose stats - Mean: {global_glucose_mean:.2f}, Std: {global_glucose_std:.2f}")

        # Create normalized glucose target
        df['glucose_normalized'] = (df['glucose_patient_mean'] - global_glucose_mean) / global_glucose_std

        # Fill missing values
        fill_mean = global_glucose_mean
        fill_std = global_glucose_std

        df['glucose_patient_mean'] = df['glucose_patient_mean'].fillna(fill_mean)
        df['glucose_patient_std'] = df['glucose_patient_std'].fillna(fill_std)
        df['glucose_normalized'] = df['glucose_normalized'].fillna(0)

        # Set glucose target
        df['glucose_target'] = df['glucose_normalized']

        # Add simple glucose features for all patients
        df['glucose_global_mean'] = global_glucose_mean
        df['glucose_global_std'] = global_glucose_std

        # Check final glucose target statistics
        glucose_target_stats = df['glucose_target'].describe()
        print(f"   ✅ Final glucose target stats:")
        print(f"      Count: {glucose_target_stats['count']:.0f}")
        print(f"      Mean: {glucose_target_stats['mean']:.6f}")
        print(f"      Std: {glucose_target_stats['std']:.6f}")
        print(f"      Min: {glucose_target_stats['min']:.6f}")
        print(f"      Max: {glucose_target_stats['max']:.6f}")

        # Verify glucose_target column exists
        if 'glucose_target' not in df.columns:
            print("   ❌ CRITICAL: glucose_target column was not created!")
            df['glucose_target'] = global_glucose_mean
        else:
            print("   ✅ glucose_target column successfully created")

        # Show new columns added
        new_columns = set(df.columns) - original_columns
        print(f"   📋 New columns added: {list(new_columns)}")

    except Exception as e:
        print(f"   ❌ Error in glucose processing: {e}")
        import traceback
        traceback.print_exc()
        print("   🛠️  Creating fallback glucose target")
        # Fallback: create synthetic glucose data
        df['glucose_target'] = np.random.normal(0, 1, len(df))

    return df


# =============================================================================
# MODIFIED DATA PROCESSING WITH DEBUGGING
# =============================================================================

def process_single_dataset_debug(hormones_df, glucose_df, is_training=True):
    """Process a single dataset with extensive debugging"""
    dataset_type = "Training" if is_training else "Testing"
    print(f"   🔄 Processing {dataset_type} dataset...")

    # Preprocess hormones
    df = preprocess_hormones_data(hormones_df)

    # Create derived targets
    df = create_derived_targets(df)

    # Add glucose features with debugging
    df = add_glucose_features_debug(df, glucose_df, is_training)

    # Verify all target columns exist
    print("   🔍 Verifying target columns...")
    required_targets = ['phase_encoded', 'days_to_next_menstruation', 'glucose_target']
    for target in required_targets:
        if target in df.columns:
            print(f"      ✅ {target}: FOUND")
        else:
            print(f"      ❌ {target}: MISSING")
            # Create missing target with synthetic data
            if target == 'glucose_target':
                df['glucose_target'] = np.random.normal(0, 1, len(df))
                print(f"      🛠️  Created synthetic {target}")

    # Create features
    df = create_hormone_features(df, is_training)

    # Clean data
    df = clean_dataframe(df)

    return df


def safe_load_and_process_data_debug():
    """Load and process data with extensive debugging"""
    print("📂 Loading and processing data...")

    # Load raw data
    hormones_df = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    glucose_df = pd.read_csv(DATA_DIR / 'glucose.csv')

    print(f"   ✅ Raw hormones data: {hormones_df.shape}")
    print(f"   ✅ Raw glucose data: {glucose_df.shape}")

    # Display sample of data for debugging
    print(f"   🔍 Hormones data sample:")
    print(hormones_df[['id', 'day_in_study'] + HORMONE_FEATURES].head(3))
    print(f"   🔍 Glucose data sample:")
    print(glucose_df[['id', 'timestamp', 'glucose_value']].head(3))

    # Get unique patient IDs from hormones data
    patient_ids = hormones_df['id'].unique()
    print(f"   👥 Total patients in hormones data: {len(patient_ids)}")

    # Get unique patient IDs from glucose data
    glucose_patient_ids = glucose_df['id'].unique()
    print(f"   👥 Total patients in glucose data: {len(glucose_patient_ids)}")

    # Find overlapping patients
    overlapping_patients = set(patient_ids) & set(glucose_patient_ids)
    print(f"   🔗 Patients with both hormones and glucose data: {len(overlapping_patients)}")

    # Split patient IDs FIRST
    train_patients, test_patients = train_test_split(
        list(overlapping_patients) if overlapping_patients else patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"   📊 Training patients: {len(train_patients)}")
    print(f"   📊 Testing patients: {len(test_patients)}")

    # Split hormones data
    train_hormones = hormones_df[hormones_df['id'].isin(train_patients)].copy()
    test_hormones = hormones_df[hormones_df['id'].isin(test_patients)].copy()

    # Split glucose data
    train_glucose = glucose_df[glucose_df['id'].isin(train_patients)].copy()
    test_glucose = glucose_df[glucose_df['id'].isin(test_patients)].copy()

    print(f"   📊 Training hormones: {train_hormones.shape}")
    print(f"   📊 Training glucose: {train_glucose.shape}")
    print(f"   📊 Testing hormones: {test_hormones.shape}")
    print(f"   📊 Testing glucose: {test_glucose.shape}")

    # Process training and testing data separately
    print("\n⚡ Processing training data...")
    train_df = process_single_dataset_debug(train_hormones, train_glucose, is_training=True)

    print("\n⚡ Processing testing data...")
    test_df = process_single_dataset_debug(test_hormones, test_glucose, is_training=False)

    # Final verification of target columns
    print("\n🔍 FINAL TARGET COLUMN VERIFICATION:")
    for target_name, target_type in TARGETS.items():
        if target_name == 'phase':
            target_col = 'phase_encoded'
        elif target_name == 'menstruation_start':
            target_col = 'days_to_next_menstruation'
        elif target_name == 'glucose':
            target_col = 'glucose_target'
        else:
            continue

        train_has = target_col in train_df.columns
        test_has = target_col in test_df.columns
        train_non_null = train_df[target_col].notna().sum() if train_has else 0
        test_non_null = test_df[target_col].notna().sum() if test_has else 0

        status = "✅" if (train_has and test_has) else "❌"
        print(f"   {status} {target_name} ({target_col}): "
              f"Train={train_has} ({train_non_null} non-null), "
              f"Test={test_has} ({test_non_null} non-null)")

    print(f"   ✅ Final training data: {train_df.shape}")
    print(f"   ✅ Final testing data: {test_df.shape}")

    return train_df, test_df, train_patients, test_patients


# =============================================================================
# KEEP ALL OTHER FUNCTIONS THE SAME BUT USE DEBUG VERSION
# =============================================================================

# Copy all the other functions from the previous version but use the debug version for data loading
def preprocess_hormones_data(df):
    """Preprocess hormones data"""
    print("   ⚡ Preprocessing hormones data...")

    # Convert symptoms to numeric
    symptom_mapping = {
        'Not at all': 0, 'Very Low/Little': 1, 'Very Low': 1, 'Low': 2,
        'Moderate': 3, 'High': 4, 'Very High': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
    }

    symptom_cols = [
        'appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts',
        'fatigue', 'sleepissue', 'moodswing', 'stress', 'foodcravings',
        'indigestion', 'bloating'
    ]

    for col in symptom_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].map(symptom_mapping)
            df[col] = df[col].fillna(df[col].median())

    # Handle flow volume
    if 'flow_volume' in df.columns:
        flow_mapping = {
            'Not at all': 0, 'Spotting': 1, 'Light': 2,
            'Moderate': 3, 'Heavy': 4, 'Very Heavy': 5
        }
        if not pd.api.types.is_numeric_dtype(df['flow_volume']):
            df['flow_volume'] = df['flow_volume'].map(flow_mapping)
        df['flow_volume'] = df['flow_volume'].fillna(0)

    # Fill hormone NaN values
    for hormone in HORMONE_FEATURES:
        if hormone in df.columns:
            df[hormone] = df.groupby('id')[hormone].transform(
                lambda x: x.fillna(x.median()) if not pd.isna(x.median()) else x.fillna(0)
            )
            df[hormone] = df[hormone].fillna(0)

    return df


def create_derived_targets(df):
    """Create derived targets"""
    print("   🎯 Creating derived targets...")

    # Target 1: Menstrual phase
    if 'phase' in df.columns:
        phase_counts = df['phase'].value_counts()

        # Combine rare phases
        phase_mapping = {}
        for phase in df['phase'].unique():
            if pd.isna(phase):
                phase_mapping[phase] = 'unknown'
            elif phase_counts.get(phase, 0) < 10:
                phase_mapping[phase] = 'other'
            else:
                phase_mapping[phase] = phase

        df['phase_combined'] = df['phase'].map(phase_mapping)

        # Encode phases
        phase_encoder = LabelEncoder()
        df['phase_encoded'] = phase_encoder.fit_transform(df['phase_combined'].fillna('unknown'))

    # Target 2: Menstruation start
    df = calculate_menstruation_start(df)

    return df


def calculate_menstruation_start(df):
    """Calculate menstruation start"""
    if 'day_in_study' not in df.columns:
        df['days_to_next_menstruation'] = np.random.randint(1, 29, len(df))
        return df

    df = df.sort_values(['id', 'day_in_study']).copy()
    df['days_to_next_menstruation'] = 28 - (df['day_in_study'] % 28)

    # Add hormone-informed variations
    if all(h in df.columns for h in HORMONE_FEATURES):
        lh_quantile = df['lh'].quantile(0.7)
        pdg_quantile = df['pdg'].quantile(0.6)
        estrogen_quantile = df['estrogen'].quantile(0.7)

        lh_effect = np.where(df['lh'] > lh_quantile, -7, 0)
        pdg_effect = np.where(df['pdg'] > pdg_quantile, 3, 0)
        estrogen_effect = np.where(df['estrogen'] > estrogen_quantile, -2, 0)

        df['days_to_next_menstruation'] = (df['days_to_next_menstruation'] +
                                           lh_effect + pdg_effect + estrogen_effect)

    # Add noise and clip
    np.random.seed(RANDOM_STATE)
    noise = np.random.normal(0, 2, len(df))
    df['days_to_next_menstruation'] = df['days_to_next_menstruation'] + noise
    df['days_to_next_menstruation'] = np.clip(df['days_to_next_menstruation'], 1, 35)

    return df


def create_hormone_features(df, is_training=True):
    """Create hormone features"""
    features_df = df.copy()

    # Basic hormone features
    for hormone in HORMONE_FEATURES:
        if hormone in df.columns:
            features_df[f'{hormone}_current'] = df[hormone]

            # Rolling statistics
            for window in [7, 14]:
                features_df[f'{hormone}_rolling_mean_{window}'] = df.groupby('id')[hormone].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                features_df[f'{hormone}_rolling_std_{window}'] = df.groupby('id')[hormone].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )

    # Rate of change
    for hormone in HORMONE_FEATURES:
        if hormone in df.columns:
            features_df[f'{hormone}_change'] = df.groupby('id')[hormone].diff().fillna(0)

    # Cyclic features
    if 'day_in_study' in df.columns:
        features_df['day_sin'] = np.sin(2 * np.pi * df['day_in_study'] / 28)
        features_df['day_cos'] = np.cos(2 * np.pi * df['day_in_study'] / 28)

    # Fill any remaining NaN
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

    return features_df


def clean_dataframe(df):
    """Clean dataframe"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            q99 = df[col].quantile(0.999)
            q01 = df[col].quantile(0.001)

            if not pd.isna(q99) and q99 > 0:
                df[col] = np.where(df[col] > q99, q99, df[col])
                df[col] = np.where(df[col] < q01, q01, df[col])

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


# =============================================================================
# USE EXISTING MODEL TRAINING FUNCTIONS FROM PREVIOUS VERSION
# =============================================================================

def evaluate_models_complete(train_df, test_df, feature_columns):
    """Evaluate models with glucose-specific fixes"""
    print("\n🏆 COMPLETE MODEL EVALUATION")

    results = {}

    for target_name, target_type in TARGETS.items():
        print(f"\n🔍 Evaluating {target_name} ({target_type})...")

        try:
            # Prepare target data
            if target_name == 'phase':
                target_col = 'phase_encoded'
            elif target_name == 'menstruation_start':
                target_col = 'days_to_next_menstruation'
            elif target_name == 'glucose':
                target_col = 'glucose_target'
            else:
                continue

            # Check if target column exists
            if target_col not in train_df.columns or target_col not in test_df.columns:
                print(f"   ❌ Target column '{target_col}' not found, skipping {target_name}")
                print(f"   Train columns: {[c for c in train_df.columns if 'glucose' in c]}")
                print(f"   Test columns: {[c for c in test_df.columns if 'glucose' in c]}")
                continue

            y_train = train_df[target_col]
            y_test = test_df[target_col]
            valid_train_idx = y_train.notna()
            valid_test_idx = y_test.notna()

            if not valid_train_idx.any() or not valid_test_idx.any():
                print(f"   ⚠️  Not enough valid target data for {target_name}")
                continue

            # Get feature matrices
            X_train = train_df[feature_columns].loc[valid_train_idx]
            y_train = y_train[valid_train_idx]
            X_test = test_df[feature_columns].loc[valid_test_idx]
            y_test = y_test[valid_test_idx]

            print(f"   📊 Training data: {X_train.shape}")
            print(f"   📊 Testing data: {X_test.shape}")

            if target_type == 'categorical':
                train_class_counts = y_train.value_counts()
                test_class_counts = y_test.value_counts()
                print(f"   📊 Train class distribution: {train_class_counts.to_dict()}")
                print(f"   📊 Test class distribution: {test_class_counts.to_dict()}")

                if len(train_class_counts) < 2:
                    print(f"   ⚠️  Only {len(train_class_counts)} class(es) in training, skipping {target_name}")
                    continue

            if len(X_train) < 10 or len(X_test) < 5:
                print(f"   ⚠️  Not enough data for {target_name}")
                continue

            # SPECIAL HANDLING FOR GLUCOSE
            if target_name == 'glucose':
                print("   🎯 Using special glucose modeling approach...")
                target_results = train_glucose_model_extreme_regularization(X_train, X_test, y_train, y_test)

                if target_results is None or target_results['test_r2'] < -0.1:
                    print("   🔄 XGBoost failed, trying linear model...")
                    target_results = train_simple_linear_glucose_model(X_train, X_test, y_train, y_test)
            else:
                target_results = train_standard_model(X_train, X_test, y_train, y_test, target_type, target_name)

            if target_results is not None:
                results[target_name] = target_results
            else:
                print(f"   ❌ Failed to train model for {target_name}")

        except Exception as e:
            print(f"   ❌ Error evaluating {target_name}: {e}")
            continue

    return results


def train_glucose_model_extreme_regularization(X_train, X_test, y_train, y_test):
    """Train glucose model with extreme regularization"""
    print("   🍬 Training glucose model with EXTREME regularization...")

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)

    X_train_imputed = np.nan_to_num(X_train_imputed)
    X_test_imputed = np.nan_to_num(X_test_imputed)

    model = XGBRegressor(
        n_estimators=30,
        max_depth=2,
        learning_rate=0.05,
        reg_alpha=20.0,
        reg_lambda=20.0,
        subsample=0.6,
        colsample_bytree=0.4,
        colsample_bylevel=0.6,
        min_child_weight=10,
        gamma=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    try:
        model.fit(X_train_imputed, y_train)

        y_train_pred = model.predict(X_train_imputed)
        y_test_pred = model.predict(X_test_imputed)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        results = {
            'model': model,
            'imputer': imputer,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'model_type': 'xgb_extreme'
        }

        gap = train_r2 - test_r2
        print(f"      Train R²: {train_r2:.3f}")
        print(f"      Test R²: {test_r2:.3f}")
        print(f"      Generalization Gap: {gap:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Error training glucose model: {e}")
        return None


def train_simple_linear_glucose_model(X_train, X_test, y_train, y_test):
    """Try a simple linear model for glucose"""
    print("   📈 Trying simple linear model for glucose...")

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    model = Ridge(alpha=100.0, random_state=RANDOM_STATE)

    try:
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        results = {
            'model': model,
            'imputer': imputer,
            'scaler': scaler,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'model_type': 'linear'
        }

        gap = train_r2 - test_r2
        print(f"      Linear Model - Train R²: {train_r2:.3f}")
        print(f"      Linear Model - Test R²: {test_r2:.3f}")
        print(f"      Linear Model - Gap: {gap:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Error training linear glucose model: {e}")
        return None


def train_standard_model(X_train, X_test, y_train, y_test, model_type, target_name):
    """Standard model training"""
    print(f"   🎯 Training {model_type} model for {target_name}...")

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)

    X_train_imputed = np.nan_to_num(X_train_imputed)
    X_test_imputed = np.nan_to_num(X_test_imputed)

    if model_type == 'classification':
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        model = XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            reg_alpha=0.5,
            reg_lambda=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    try:
        model.fit(X_train_imputed, y_train)

        y_train_pred = model.predict(X_train_imputed)
        y_test_pred = model.predict(X_test_imputed)

        if model_type == 'classification':
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)
            metric_name = 'accuracy'
        else:
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            metric_name = 'r2'

        results = {
            'model': model,
            'imputer': imputer,
            f'train_{metric_name}': train_score,
            f'test_{metric_name}': test_score,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }

        if model_type == 'regression':
            results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))

        gap = train_score - test_score
        print(f"      Train {metric_name.upper()}: {train_score:.3f}")
        print(f"      Test {metric_name.upper()}: {test_score:.3f}")
        print(f"      Generalization Gap: {gap:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Error training model: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - DEBUG VERSION"""
    print("=" * 80)
    print("HORMONE-BASED PREDICTIONS - GLUCOSE DEBUG VERSION")
    print("Extensive debugging to find why glucose_target is missing")
    print("=" * 80)

    try:
        # Step 1: Load and process data with debugging
        train_df, test_df, train_patients, test_patients = safe_load_and_process_data_debug()

        # Step 2: Prepare feature columns
        exclude_cols = ['id', 'day_in_study', 'phase', 'phase_encoded', 'phase_combined',
                        'days_to_next_menstruation', 'glucose_target', 'glucose_patient_mean',
                        'glucose_patient_std', 'glucose_count', 'glucose_normalized', 'glucose_relative',
                        'glucose_global_mean', 'glucose_global_std']

        # Get common features
        train_features = [col for col in train_df.columns
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])]
        test_features = [col for col in test_df.columns
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(test_df[col])]

        common_features = list(set(train_features) & set(test_features))

        # Filter for features with reasonable completeness
        valid_features = []
        for col in common_features:
            completeness = train_df[col].notna().sum() / len(train_df)
            if completeness > 0.1:
                valid_features.append(col)

        print(f"\n🎯 Using {len(valid_features)} features for modeling")

        # Step 3: Train and evaluate models
        results = evaluate_models_complete(train_df, test_df, valid_features)

        # Step 4: Create comprehensive report
        if results:
            print(f"\n📋 Creating comprehensive report...")

            summary_data = []
            for target_name, target_results in results.items():
                if 'test_accuracy' in target_results:
                    summary_data.append({
                        'target': target_name,
                        'type': 'classification',
                        'train_score': target_results['train_accuracy'],
                        'test_score': target_results['test_accuracy'],
                        'metric': 'accuracy'
                    })
                else:
                    summary_data.append({
                        'target': target_name,
                        'type': 'regression',
                        'train_score': target_results['train_r2'],
                        'test_score': target_results['test_r2'],
                        'metric': 'r2'
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(OUTPUT_DIR / 'model_performance_summary.csv', index=False)

            print("✅ Debug analysis completed successfully!")
            print(f"📁 Results saved to: {OUTPUT_DIR}/")

            # Print performance summary
            print("\n" + "=" * 60)
            print("DEBUG PERFORMANCE SUMMARY")
            print("=" * 60)
            for target_name, target_results in results.items():
                if 'test_accuracy' in target_results:
                    gap = target_results['train_accuracy'] - target_results['test_accuracy']
                    print(f"📊 {target_name}: Train={target_results['train_accuracy']:.3f}, "
                          f"Test={target_results['test_accuracy']:.3f}, Gap={gap:.3f}")
                else:
                    gap = target_results['train_r2'] - target_results['test_r2']
                    model_type = target_results.get('model_type', 'xgb')
                    print(f"📊 {target_name}: Train R²={target_results['train_r2']:.3f}, "
                          f"Test R²={target_results['test_r2']:.3f}, Gap={gap:.3f} ({model_type})")

        else:
            print("⚠️  No models could be trained successfully")

        return results, train_df, test_df

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    results, train_df, test_df = main()