# run_complete_analysis.py
"""
COMPLETE PLS ANALYSIS PIPELINE FOR HORMONE PREDICTION
Run this file to execute the entire analysis from start to finish.
"""

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS IF NEEDED
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('pls_analysis_report')
REQUIRED_FILES = [
    'hormones_and_selfreport.csv',
    'glucose.csv',
    'sleep.csv'
]

OPTIONAL_FILES = [
    'active_minutes.csv',
    'heart_rate_variability_details.csv',
    'resting_heart_rate.csv',
    'stress_score.csv',
    'steps.csv',
    'computed_temperature.csv'
]


# =============================================================================
# STEP 1: DATA LOADING AND VALIDATION
# =============================================================================

def check_required_files():
    """Verify all required files exist"""
    print("🔍 Checking for required files...")
    missing_files = []
    for file in REQUIRED_FILES:
        if not (DATA_DIR / file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files found!")
        return True


def load_data_files():
    """Load all available data files"""
    print("\n📂 Loading data files...")

    # Load required files
    hormones = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(DATA_DIR / 'glucose.csv')
    sleep = pd.read_csv(DATA_DIR / 'sleep.csv')

    print(f"   hormones_and_selfreport.csv: {hormones.shape}")
    print(f"   glucose.csv: {glucose.shape}")
    print(f"   sleep.csv: {sleep.shape}")

    # Load optional files if available
    optional_data = {}
    for file in OPTIONAL_FILES:
        file_path = DATA_DIR / file
        if file_path.exists():
            optional_data[file] = pd.read_csv(file_path)
            print(f"   {file}: {optional_data[file].shape}")
        else:
            print(f"   {file}: Not found (optional)")

    return hormones, glucose, sleep, optional_data


# =============================================================================
# STEP 2: DATA PROCESSING AND INTEGRATION
# =============================================================================

def integrate_data(hormones, glucose, sleep, optional_data):
    """Integrate datasets using daily aggregates to avoid memory issues"""
    print("\n🔄 Integrating datasets with daily aggregation...")

    # Process glucose data (aggregate to daily level)
    print("   Processing glucose data...")
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std', 'min', 'max']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max']

    # Start with core merge
    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    merged_data = pd.merge(merged_data, sleep,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')

    print(f"   After core merge: {merged_data.shape}")

    # Process and merge optional datasets with AGGREGATION
    for file_name, df in optional_data.items():
        dataset_key = file_name.replace('.csv', '')
        try:
            print(f"   Processing {file_name}...")

            # Handle different dataset types with aggregation
            if file_name == 'active_minutes.csv':
                # Already daily data, just merge
                merged_data = pd.merge(merged_data, df,
                                       on=['id', 'day_in_study'],
                                       how='left',
                                       suffixes=('', f'_{dataset_key}'))

            elif file_name == 'heart_rate_variability_details.csv':
                # Aggregate HRV data by day
                hrv_daily = df.groupby(['id', 'day_in_study']).agg({
                    'rmssd': ['mean', 'std'],
                    'low_frequency': 'mean',
                    'high_frequency': 'mean'
                }).reset_index()
                hrv_daily.columns = ['id', 'day_in_study', 'hrv_rmssd_mean', 'hrv_rmssd_std',
                                     'hrv_low_freq', 'hrv_high_freq']
                merged_data = pd.merge(merged_data, hrv_daily,
                                       on=['id', 'day_in_study'], how='left')

            elif file_name == 'resting_heart_rate.csv':
                # Already daily data
                merged_data = pd.merge(merged_data, df.rename(columns={'value': 'resting_hr'}),
                                       on=['id', 'day_in_study'], how='left')

            elif file_name == 'stress_score.csv':
                # Already daily data
                merged_data = pd.merge(merged_data, df,
                                       on=['id', 'day_in_study'], how='left',
                                       suffixes=('', f'_{dataset_key}'))

            elif file_name == 'steps.csv':
                # Aggregate steps by day
                steps_daily = df.groupby(['id', 'day_in_study']).agg({
                    'steps': 'sum'
                }).reset_index()
                steps_daily.columns = ['id', 'day_in_study', 'total_steps']
                merged_data = pd.merge(merged_data, steps_daily,
                                       on=['id', 'day_in_study'], how='left')

            elif file_name == 'computed_temperature.csv':
                # Use the nightly temperature
                temp_daily = df[['id', 'sleep_start_day_in_study', 'nightly_temperature']].copy()
                temp_daily = temp_daily.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
                merged_data = pd.merge(merged_data, temp_daily,
                                       on=['id', 'day_in_study'], how='left',
                                       suffixes=('', f'_{dataset_key}'))

            print(f"   Added {file_name}: {merged_data.shape}")

        except Exception as e:
            print(f"   ⚠️  Could not merge {file_name}: {e}")

    return merged_data

def convert_symptoms_to_numeric(df):
    """Convert all symptom string values to numeric Likert scale"""
    print("\n🔢 Converting symptom strings to numeric values...")

    # Comprehensive mapping for all symptom columns
    symptom_mapping = {
        'Not at all': 0,
        'Very Low/Little': 1,
        'Very Low': 1,
        'Low': 2,
        'Moderate': 3,
        'High': 4,
        'Very High': 5,
        # Handle any numeric strings
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }

    symptom_columns = [
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating'
    ]

    for col in symptom_columns:
        if col in df.columns:
            non_missing_before = df[col].notna().sum()
            df[col] = df[col].map(symptom_mapping)
            non_missing_after = df[col].notna().sum()
            if non_missing_before != non_missing_after:
                print(f"   {col}: {non_missing_before - non_missing_after} values couldn't be mapped")

    # Convert other categorical columns
    if 'flow_volume' in df.columns:
        flow_mapping = {'Not at all': 0, 'Spotting / Very Light': 1, 'Light': 2,
                        'Somewhat Light': 2, 'Moderate': 3, 'Somewhat Heavy': 4,
                        'Heavy': 5, 'Very Heavy': 6}
        df['flow_volume'] = df['flow_volume'].map(flow_mapping)

    if 'phase' in df.columns:
        df['phase'] = df['phase'].astype('category').cat.codes.replace(-1, np.nan)

    # Convert boolean columns
    bool_columns = ['is_weekend_y', 'mainsleep', 'is_weekend']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0})

    return df


# =============================================================================
# STEP 3: PLS ANALYSIS
# =============================================================================

def run_pls_analysis(merged_data):
    """Run the complete PLS analysis"""
    print("\n🧪 Running PLS analysis...")

    # Define predictors and targets
    predictors = [
        # Symptoms
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating',
        # Glucose metrics
        'glucose_mean', 'glucose_std',
        # Sleep metrics
        'minutesasleep', 'efficiency',
        # Optional metrics (will be filtered if not available)
        'sedentary', 'lightly', 'moderately', 'very',  # from active_minutes
        'rmssd', 'low_frequency', 'high_frequency',  # from HRV
        'value',  # from resting_heart_rate
        'stress_score'  # from stress_score
    ]

    targets = ['lh', 'estrogen', 'pdg']

    # Filter to available columns
    available_predictors = [col for col in predictors if col in merged_data.columns]
    available_targets = [col for col in targets if col in merged_data.columns]

    print(f"   Using {len(available_predictors)} predictors: {available_predictors}")
    print(f"   Predicting {len(available_targets)} hormones: {available_targets}")

    # Prepare analysis data
    analysis_data = merged_data[available_predictors + available_targets].copy()

    # Drop rows with missing targets
    analysis_data = analysis_data.dropna(subset=available_targets)
    print(f"   Complete cases: {analysis_data.shape[0]}")

    if analysis_data.shape[0] < 10:
        print("❌ Not enough complete cases for analysis")
        return None

    X = analysis_data[available_predictors]
    Y = analysis_data[available_targets]

    # Filter to columns with data
    columns_with_data = [col for col in X.columns if X[col].notna().sum() > 0]
    X = X[columns_with_data]
    print(f"   Predictors with data: {len(columns_with_data)}")

    # Handle missing values and standardize
    imputer = SimpleImputer(strategy='median')
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler_X.fit_transform(X_imputed)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Run PLS regression
    n_components = min(3, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    # Calculate performance
    Y_pred_scaled = pls.predict(X_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    r2_scores = {}
    for i, hormone in enumerate(available_targets):
        r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
        r2_scores[hormone] = r2

    # Calculate feature importance (VIP scores)
    W = pls.x_weights_
    Q = pls.y_loadings_

    p = W.shape[0]
    SS_weights = np.sum(W ** 2, axis=0)
    R2Y = np.sum(Q ** 2, axis=0)
    total_R2Y = np.sum(R2Y)

    VIP_scores = np.sqrt(p * np.sum((W ** 2) * (R2Y / total_R2Y), axis=1))

    importance_df = pd.DataFrame({
        'feature': columns_with_data,
        'VIP_score': VIP_scores
    }).sort_values('VIP_score', ascending=False)

    # Compile results
    results = {
        'pls_model': pls,
        'importance_df': importance_df,
        'r2_scores': r2_scores,
        'X': pd.DataFrame(X_imputed, columns=columns_with_data, index=X.index),
        'Y': Y,
        'predictors': columns_with_data,
        'targets': available_targets,
        'n_components': n_components,
        'n_observations': X.shape[0]
    }

    return results


# =============================================================================
# STEP 4: REPORT GENERATION
# =============================================================================

def generate_report(results, merged_data):
    """Generate comprehensive analysis report"""
    print("\n📊 Generating analysis report...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate text report
    report_content = generate_text_report(results, merged_data)

    # Save text report
    with open(OUTPUT_DIR / 'pls_analysis_report.txt', 'w') as f:
        f.write(report_content)

    # Generate visualizations
    generate_visualizations(results, merged_data)

    # Save data files
    generate_data_files(results)

    print(f"✅ Report generated in '{OUTPUT_DIR}' directory")
    return report_content


def generate_text_report(results, data):
    """Generate the main text report"""
    report = []
    report.append("=" * 80)
    report.append("PLS ANALYSIS REPORT - HORMONE PREDICTION")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("1. EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Observations: {results['n_observations']}")
    report.append(f"Predictors: {len(results['predictors'])}")
    report.append(f"Target Hormones: {len(results['targets'])}")
    report.append(f"PLS Components: {results['n_components']}")
    report.append("")

    # Model Performance
    report.append("2. MODEL PERFORMANCE")
    report.append("-" * 40)
    report.append("R² Scores (Variance Explained):")
    for hormone, r2 in results['r2_scores'].items():
        report.append(f"  {hormone.upper():<10} : {r2:.4f}")
    report.append("")

    # Feature Importance
    report.append("3. TOP 10 MOST IMPORTANT FEATURES")
    report.append("-" * 40)
    for i, (idx, row) in enumerate(results['importance_df'].head(10).iterrows()):
        importance = "***" if row['VIP_score'] > 1.0 else "** " if row['VIP_score'] > 0.8 else "*  "
        report.append(f"  {i + 1:2d}. {importance} {row['feature']:<25} : {row['VIP_score']:.4f}")
    report.append("")
    report.append("VIP > 1.0: Highly important, VIP 0.8-1.0: Moderately important, VIP < 0.8: Less important")
    report.append("")

    # Data Quality
    report.append("4. DATA QUALITY")
    report.append("-" * 40)
    report.append("Predictor Completeness:")
    for predictor in results['predictors']:
        non_missing = data[predictor].notna().sum()
        pct = (non_missing / len(data)) * 100
        report.append(f"  {predictor:<25} : {non_missing:>4} / {len(data):>4} ({pct:5.1f}%)")

    return "\n".join(report)


def generate_visualizations(results, data):
    """Generate visualization plots"""
    # Feature importance plot
    plt.figure(figsize=(12, 8))

    # Top features plot
    plt.subplot(2, 2, 1)
    top_features = results['importance_df'].head(15)
    colors = ['red' if score > 1.0 else 'orange' if score > 0.8 else 'blue'
              for score in top_features['VIP_score']]
    plt.barh(range(len(top_features)), top_features['VIP_score'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('VIP Score')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()

    # Performance plot
    plt.subplot(2, 2, 2)
    hormones = list(results['r2_scores'].keys())
    r2_scores = [results['r2_scores'][h] for h in hormones]
    bars = plt.bar(hormones, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylim(0, 1)
    plt.ylabel('R² Score')
    plt.title('Model Performance by Hormone')
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_data_files(results):
    """Generate CSV data files"""
    results['importance_df'].to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    performance_df = pd.DataFrame([
        {'hormone': hormone, 'r_squared': r2}
        for hormone, r2 in results['r2_scores'].items()
    ])
    performance_df.to_csv(OUTPUT_DIR / 'model_performance.csv', index=False)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete analysis"""
    print("=" * 70)
    print("PLS ANALYSIS FOR HORMONE PREDICTION")
    print("=" * 70)

    # Step 1: Check and load files
    if not check_required_files():
        return

    hormones, glucose, sleep, optional_data = load_data_files()

    # Step 2: Process and integrate data
    merged_data = integrate_data(hormones, glucose, sleep, optional_data)
    merged_data = convert_symptoms_to_numeric(merged_data)

    # Step 3: Run PLS analysis
    results = run_pls_analysis(merged_data)

    if results is None:
        print("❌ Analysis failed - not enough data")
        return

    # Step 4: Generate report
    report = generate_report(results, merged_data)

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Observations: {results['n_observations']}")
    print(f"Predictors used: {len(results['predictors'])}")
    print("R² Scores:")
    for hormone, r2 in results['r2_scores'].items():
        print(f"  {hormone}: {r2:.3f}")
    print(f"\nReport saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()