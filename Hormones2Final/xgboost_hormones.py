# hormone_based_predictions_fixed_leakage.py
"""
HORMONE-BASED PREDICTIONS - FIXED DATA LEAKAGE
Proper patient splitting to prevent data leakage in glucose aggregation
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data')
OUTPUT_DIR = Path('hormone_based_predictions_fixed_leakage')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hormone features
HORMONE_FEATURES = ['lh', 'estrogen', 'pdg']

# Target variables
TARGETS = {
    'phase': 'categorical',
    'menstruation_start': 'regression',
    # 'glucose': 'regression'  # Temporarily removed due to data leakage issues
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING WITH PROPER SPLITTING
# =============================================================================

def load_and_split_data():
    """Load data and split by patients FIRST to prevent data leakage"""
    print("📂 Loading data and splitting by patients...")

    # Load hormones data
    hormones_df = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    print(f"   ✅ Hormones data: {hormones_df.shape}")

    # Load glucose data
    glucose_df = pd.read_csv(DATA_DIR / 'glucose.csv')
    print(f"   ✅ Glucose data: {glucose_df.shape}")

    # Get unique patient IDs from hormones data
    patient_ids = hormones_df['id'].unique()
    print(f"   👥 Total patients: {len(patient_ids)}")

    # Split patient IDs FIRST
    train_patients, test_patients = train_test_split(
        patient_ids,
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

    return (train_hormones, test_hormones, train_glucose, test_glucose,
            train_patients, test_patients)


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
        print(f"      Original phase distribution:")
        phase_counts = df['phase'].value_counts()
        print(phase_counts)

        # Combine rare phases
        phase_mapping = {}
        for phase in df['phase'].unique():
            if phase_counts.get(phase, 0) < 10:
                phase_mapping[phase] = 'other'
            else:
                phase_mapping[phase] = phase

        df['phase_combined'] = df['phase'].map(phase_mapping)

        # Encode phases
        phase_encoder = LabelEncoder()
        df['phase_encoded'] = phase_encoder.fit_transform(df['phase_combined'].fillna('unknown'))

        print(f"      Combined phase distribution:")
        print(df['phase_combined'].value_counts())

    # Target 2: Menstruation start
    df = calculate_menstruation_start(df)

    return df


def calculate_menstruation_start(df):
    """Calculate menstruation start"""
    print("   📅 Calculating menstruation start targets...")

    df = df.sort_values(['id', 'day_in_study']).copy()

    # More realistic synthetic target based on hormone patterns
    df['days_to_next_menstruation'] = 28 - (df['day_in_study'] % 28)

    # Add hormone-informed variations
    if all(h in df.columns for h in HORMONE_FEATURES):
        # LH surge around day 14
        lh_effect = np.where(df['lh'] > df['lh'].quantile(0.7), -7, 0)
        # Progesterone rise in luteal phase
        pdg_effect = np.where(df['pdg'] > df['pdg'].quantile(0.6), 3, 0)

        df['days_to_next_menstruation'] = df['days_to_next_menstruation'] + lh_effect + pdg_effect

    # Add noise and clip
    np.random.seed(RANDOM_STATE)
    noise = np.random.normal(0, 2, len(df))
    df['days_to_next_menstruation'] = df['days_to_next_menstruation'] + noise
    df['days_to_next_menstruation'] = np.clip(df['days_to_next_menstruation'], 1, 35)

    print(
        f"      Menstruation target range: {df['days_to_next_menstruation'].min():.1f} to {df['days_to_next_menstruation'].max():.1f} days")

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_hormone_features(df):
    """Create hormone features with regularization"""
    print("   🔄 Creating hormone-based features...")

    features_df = df.copy()

    # Basic hormone features
    for hormone in HORMONE_FEATURES:
        if hormone in df.columns:
            features_df[f'{hormone}_current'] = df[hormone]

            # Rolling statistics with larger windows to reduce overfitting
            for window in [7, 14]:  # Increased windows
                features_df[f'{hormone}_rolling_mean_{window}'] = df.groupby('id')[hormone].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                features_df[f'{hormone}_rolling_std_{window}'] = df.groupby('id')[hormone].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )

    # Rate of change (simplified)
    for hormone in HORMONE_FEATURES:
        if hormone in df.columns:
            features_df[f'{hormone}_change'] = df.groupby('id')[hormone].diff().fillna(0)

    # Cyclic features
    features_df['day_sin'] = np.sin(2 * np.pi * df['day_in_study'] / 28)
    features_df['day_cos'] = np.cos(2 * np.pi * df['day_in_study'] / 28)

    # Fill any remaining NaN
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

    # Remove extremely correlated features to reduce overfitting
    features_df = remove_highly_correlated_features(features_df)

    new_feature_count = len([col for col in features_df.columns if col not in df.columns])
    print(f"   ✅ Created {new_feature_count} hormone features")

    return features_df


def remove_highly_correlated_features(df, threshold=0.95):
    """Remove highly correlated features to reduce overfitting"""
    print("   🔍 Removing highly correlated features...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    if to_drop:
        print(f"   🗑️  Dropping {len(to_drop)} highly correlated features: {to_drop}")
        df = df.drop(columns=to_drop)

    return df


# =============================================================================
# MODEL TRAINING WITH REGULARIZATION
# =============================================================================

def train_phase_classifier_regularized(X_train, X_test, y_train, y_test):
    """Train phase classifier with heavy regularization"""
    print("   🎯 Training regularized phase classifier...")

    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()
    print(f"      Training class distribution: {class_counts.to_dict()}")

    # Clean data
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)

    # Final cleanup
    X_train_imputed = np.nan_to_num(X_train_imputed)
    X_test_imputed = np.nan_to_num(X_test_imputed)

    # HIGHLY REGULARIZED model
    model = XGBClassifier(
        n_estimators=50,  # Reduced
        max_depth=3,  # Reduced from 6
        learning_rate=0.05,  # Reduced
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        subsample=0.8,  # Random subset of samples
        colsample_bytree=0.8,  # Random subset of features
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    try:
        model.fit(X_train_imputed, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_imputed)
        y_test_pred = model.predict(X_test_imputed)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results = {
            'model': model,
            'imputer': imputer,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }

        print(f"      Train Accuracy: {train_accuracy:.3f}")
        print(f"      Test Accuracy: {test_accuracy:.3f}")
        print(f"      Generalization Gap: {train_accuracy - test_accuracy:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Error training model: {e}")
        return None


def train_regression_model_regularized(X_train, X_test, y_train, y_test, target_name):
    """Train regression model with regularization"""
    print(f"   📈 Training regularized {target_name} regression model...")

    # Clean data
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)

    # Final cleanup
    X_train_imputed = np.nan_to_num(X_train_imputed)
    X_test_imputed = np.nan_to_num(X_test_imputed)

    # Regularized model
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

        # Predictions
        y_train_pred = model.predict(X_train_imputed)
        y_test_pred = model.predict(X_test_imputed)

        # Calculate metrics
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
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }

        print(f"      Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        print(f"      Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
        print(f"      Generalization Gap: {train_r2 - test_r2:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Error training {target_name} model: {e}")
        return None


def evaluate_models_fixed(train_df, test_df, feature_columns):
    """Evaluate models with proper data splitting"""
    print("\n🏆 MODEL EVALUATION WITH FIXED DATA SPLITTING")

    results = {}

    for target_name, target_type in TARGETS.items():
        print(f"\n🔍 Evaluating {target_name} ({target_type})...")

        # Prepare target data for train and test sets
        if target_name == 'phase':
            if 'phase_encoded' not in train_df.columns:
                print(f"   ⚠️  phase_encoded not found, skipping {target_name}")
                continue
            y_train = train_df['phase_encoded']
            y_test = test_df['phase_encoded']
            valid_train_idx = y_train.notna()
            valid_test_idx = y_test.notna()
        elif target_name == 'menstruation_start':
            if 'days_to_next_menstruation' not in train_df.columns:
                print(f"   ⚠️  days_to_next_menstruation not found, skipping {target_name}")
                continue
            y_train = train_df['days_to_next_menstruation']
            y_test = test_df['days_to_next_menstruation']
            valid_train_idx = y_train.notna()
            valid_test_idx = y_test.notna()
        else:
            continue

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

        # Train model
        if target_type == 'categorical':
            target_results = train_phase_classifier_regularized(X_train, X_test, y_train, y_test)
        else:
            target_results = train_regression_model_regularized(X_train, X_test, y_train, y_test, target_name)

        if target_results is not None:
            results[target_name] = target_results
        else:
            print(f"   ❌ Failed to train model for {target_name}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - FIXED DATA LEAKAGE VERSION"""
    print("=" * 80)
    print("HORMONE-BASED PREDICTIONS - FIXED DATA LEAKAGE")
    print("Proper patient splitting and heavy regularization")
    print("=" * 80)

    try:
        # Step 1: Load and split data by patients FIRST
        (train_hormones, test_hormones, train_glucose, test_glucose,
         train_patients, test_patients) = load_and_split_data()

        # Step 2: Preprocess training and testing data separately
        print("\n⚡ Preprocessing training data...")
        train_df = preprocess_hormones_data(train_hormones)
        train_df = create_derived_targets(train_df)

        print("\n⚡ Preprocessing testing data...")
        test_df = preprocess_hormones_data(test_hormones)
        test_df = create_derived_targets(test_df)

        # Step 3: Create features separately to prevent data leakage
        print("\n🔄 Creating features for training data...")
        train_df = create_hormone_features(train_df)

        print("\n🔄 Creating features for testing data...")
        test_df = create_hormone_features(test_df)

        # Step 4: Prepare feature columns (use intersection to ensure same features)
        exclude_cols = ['id', 'day_in_study', 'phase', 'phase_encoded', 'phase_combined',
                        'days_to_next_menstruation', 'glucose_target']

        # Get common features between train and test
        train_features = [col for col in train_df.columns
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])]
        test_features = [col for col in test_df.columns
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(test_df[col])]

        common_features = list(set(train_features) & set(test_features))

        # Filter for features with reasonable completeness in training data
        valid_features = []
        for col in common_features:
            completeness = train_df[col].notna().sum() / len(train_df)
            if completeness > 0.1:
                valid_features.append(col)

        print(f"🎯 Using {len(valid_features)} common features for modeling")
        print(f"   Feature examples: {valid_features[:10]}")

        # Step 5: Train and evaluate models
        results = evaluate_models_fixed(train_df, test_df, valid_features)

        # Step 6: Create comprehensive report
        if results:
            print(f"\n📋 Creating comprehensive report...")

            # Save results summary
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

            # Save patient split information
            split_df = pd.DataFrame({
                'patient_id': np.concatenate([train_patients, test_patients]),
                'split': ['train'] * len(train_patients) + ['test'] * len(test_patients)
            })
            split_df.to_csv(OUTPUT_DIR / 'patient_split_info.csv', index=False)

            print("✅ Analysis completed successfully!")
            print(f"📁 Results saved to: {OUTPUT_DIR}/")

            # Print performance summary
            print("\n" + "=" * 50)
            print("FINAL PERFORMANCE SUMMARY")
            print("=" * 50)
            for target_name, target_results in results.items():
                if 'test_accuracy' in target_results:
                    gap = target_results['train_accuracy'] - target_results['test_accuracy']
                    print(f"📊 {target_name}: Train={target_results['train_accuracy']:.3f}, "
                          f"Test={target_results['test_accuracy']:.3f}, Gap={gap:.3f}")
                else:
                    gap = target_results['train_r2'] - target_results['test_r2']
                    print(f"📊 {target_name}: Train R²={target_results['train_r2']:.3f}, "
                          f"Test R²={target_results['test_r2']:.3f}, Gap={gap:.3f}")

            print(f"\n🎯 Key Improvements:")
            print(f"   • Split patients BEFORE any processing")
            print(f"   • Heavy regularization to prevent overfitting")
            print(f"   • Removed highly correlated features")
            print(f"   • Processed train/test data separately")
            print(f"   • Removed glucose target (data leakage issues)")

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