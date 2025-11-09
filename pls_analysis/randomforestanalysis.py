# random_forest_xgboost_analysis.py
"""
RANDOM FOREST & XGBOOST ANALYSIS FOR HORMONE PREDICTION
Fixed version with robust data merging
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('tree_models_report')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING & PREPROCESSING - FIXED VERSION
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess data with robust merging"""
    print("📂 Loading and preprocessing data...")

    # Load core datasets
    hormones = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(DATA_DIR / 'glucose.csv')
    sleep = pd.read_csv(DATA_DIR / 'sleep.csv')

    print(f"   hormones: {hormones.shape}, glucose: {glucose.shape}, sleep: {sleep.shape}")

    # Process glucose data
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std', 'min', 'max']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max']

    # Start with hormones as base
    merged_data = hormones.copy()

    # Merge glucose data
    merged_data = safe_merge(merged_data, glucose_daily, ['id', 'day_in_study'], 'glucose')
    print(f"   After glucose merge: {merged_data.shape}")

    # Merge sleep data - handle sleep-specific keys
    sleep_renamed = sleep.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
    sleep_cols_to_use = ['id', 'day_in_study', 'minutesasleep', 'efficiency', 'minutestofallasleep', 'minutesawake']
    sleep_cols_to_use = [col for col in sleep_cols_to_use if col in sleep_renamed.columns]

    merged_data = safe_merge(merged_data, sleep_renamed[sleep_cols_to_use], ['id', 'day_in_study'], 'sleep')
    print(f"   After sleep merge: {merged_data.shape}")

    # Load and merge optional datasets one by one
    optional_files = {
        'active_minutes.csv': ['sedentary', 'lightly', 'moderately', 'very'],
        'resting_heart_rate.csv': ['value'],  # Will rename to 'resting_hr'
        'stress_score.csv': ['stress_score', 'sleep_points', 'exertion_points']
    }

    for file, columns in optional_files.items():
        file_path = DATA_DIR / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"   Processing {file}: {df.shape}")

                if file == 'resting_heart_rate.csv':
                    # Rename value to resting_hr
                    df = df.rename(columns={'value': 'resting_hr'})
                    columns = ['resting_hr']

                # Select only the columns we need
                key_cols = ['id', 'day_in_study']
                available_cols = [col for col in key_cols + columns if col in df.columns]
                df_subset = df[available_cols].copy()

                merged_data = safe_merge(merged_data, df_subset, ['id', 'day_in_study'], file.replace('.csv', ''))
                print(f"   After {file} merge: {merged_data.shape}")

            except Exception as e:
                print(f"   ⚠️  Error merging {file}: {e}")
                continue

    return merged_data


def safe_merge(left_df, right_df, on_cols, suffix):
    """Safe merge that handles duplicate columns"""
    # Find overlapping columns (excluding merge keys)
    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)
    overlap_cols = right_cols - set(on_cols)
    overlap_cols = overlap_cols.intersection(left_cols)

    # If there are overlapping columns, rename them in the right dataframe
    if overlap_cols:
        rename_dict = {col: f"{col}_{suffix}" for col in overlap_cols}
        right_df = right_df.rename(columns=rename_dict)

    # Perform the merge
    merged = pd.merge(left_df, right_df, on=on_cols, how='left')
    return merged


def convert_symptoms_to_numeric(df):
    """Convert symptom strings to numeric values"""
    print("🔢 Converting symptoms to numeric...")

    symptom_mapping = {
        'Not at all': 0, 'Very Low/Little': 1, 'Very Low': 1, 'Low': 2,
        'Moderate': 3, 'High': 4, 'Very High': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }

    symptom_columns = [
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating'
    ]

    for col in symptom_columns:
        if col in df.columns:
            # Check if conversion is needed (if data is string type)
            if df[col].dtype == 'object' or any(isinstance(x, str) for x in df[col].dropna().head()):
                df[col] = df[col].map(symptom_mapping)
                print(f"   Converted {col}")

    return df


def create_interaction_features(df):
    """Create biologically relevant interaction terms"""
    print("🔄 Creating interaction features...")

    # Activity interactions
    if all(col in df.columns for col in ['lightly', 'moderately', 'very']):
        df['total_activity'] = df['lightly'] + df['moderately'] + df['very']
        df['intense_ratio'] = df['very'] / (df['lightly'] + 1)
        print("   Created activity features")

    # Stress-sleep interactions
    if all(col in df.columns for col in ['stress_score', 'efficiency']):
        df['stress_sleep_ratio'] = df['stress_score'] / (df['efficiency'] + 1)
        print("   Created stress-sleep features")

    # Glucose variability
    if all(col in df.columns for col in ['glucose_std', 'glucose_mean']):
        df['glucose_cv'] = df['glucose_std'] / (df['glucose_mean'] + 1)
        print("   Created glucose features")

    return df


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features_for_analysis(merged_data):
    """Select and prepare features for modeling"""

    # Define all potential predictors
    all_predictors = [
        # Symptoms
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating',
        # Glucose metrics
        'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max',
        # Sleep metrics
        'minutesasleep', 'efficiency', 'minutestofallasleep', 'minutesawake',
        # Activity metrics
        'sedentary', 'lightly', 'moderately', 'very',
        # Other physiological
        'resting_hr', 'stress_score',
        # Interaction terms
        'total_activity', 'intense_ratio', 'stress_sleep_ratio', 'glucose_cv'
    ]

    # Filter to available columns
    available_predictors = [col for col in all_predictors if col in merged_data.columns]
    targets = ['lh', 'estrogen', 'pdg']

    print(f"📊 Available predictors: {len(available_predictors)}")
    print(f"🎯 Targets: {targets}")

    # Select data for analysis
    analysis_data = merged_data[available_predictors + targets].copy()

    # Remove columns with too many missing values (>90% missing)
    columns_to_keep = []
    for col in available_predictors:
        non_missing = analysis_data[col].notna().sum()
        pct_missing = (1 - non_missing / len(analysis_data)) * 100
        if non_missing >= len(analysis_data) * 0.1:  # At least 10% data
            columns_to_keep.append(col)
        else:
            print(f"   Dropped {col}: {pct_missing:.1f}% missing")

    print(f"📈 Predictors after missing value filter: {len(columns_to_keep)}")

    return analysis_data, columns_to_keep, targets


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_tree_models(X, y, test_size=0.2, random_state=42):
    """Train Random Forest and XGBoost models with comprehensive evaluation"""

    # Remove rows where all targets are missing
    valid_indices = y.notna().all(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    print(f"📋 Final dataset: X={X.shape}, y={y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"🎯 Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Handle missing values for tree models
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Initialize models
    models = {
        'Random Forest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        ),
        'XGBoost': MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        )
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n🏋️ Training {model_name}...")

        # Train model
        model.fit(X_train_imputed, y_train)

        # Predictions
        y_pred = model.predict(X_test_imputed)

        # Calculate metrics for each hormone
        hormone_metrics = {}
        for i, hormone in enumerate(y.columns):
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])

            hormone_metrics[hormone] = {
                'r2': r2,
                'mse': mse,
                'mae': mae
            }

        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train_imputed, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        except:
            cv_mean = np.mean([m['r2'] for m in hormone_metrics.values()])
            cv_std = 0

        results[model_name] = {
            'model': model,
            'metrics': hormone_metrics,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'feature_names': X.columns.tolist(),
            'imputer': imputer
        }

        print(f"✅ {model_name} - CV R²: {cv_mean:.3f} ± {cv_std:.3f}")

        # Print individual hormone performance
        for hormone, metrics in hormone_metrics.items():
            print(f"   {hormone}: R² = {metrics['r2']:.3f}")

    return results, X_test_imputed, y_test


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(results, feature_names):
    """Comprehensive feature importance analysis"""
    print("\n🔍 Analyzing feature importance...")

    importance_results = {}

    for model_name, result in results.items():
        model = result['model']
        feature_importance = []

        if model_name == 'Random Forest':
            # For Random Forest - average importance across all target estimators
            for i, estimator in enumerate(model.estimators_):
                importance = estimator.feature_importances_
                feature_importance.append(importance)

            # Average across all hormones
            avg_importance = np.mean(feature_importance, axis=0)

        elif model_name == 'XGBoost':
            # For XGBoost
            for i, estimator in enumerate(model.estimators_):
                importance = estimator.feature_importances_
                feature_importance.append(importance)

            avg_importance = np.mean(feature_importance, axis=0)

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance,
            'importance_normalized': avg_importance / np.sum(avg_importance) * 100
        }).sort_values('importance', ascending=False)

        importance_results[model_name] = importance_df

        print(f"\n📊 {model_name} - Top 10 Features:")
        for i, (idx, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i + 1:2d}. {row['feature']:<25} : {row['importance_normalized']:.2f}%")

    return importance_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_visualizations(results, importance_results, X_test, y_test):
    """Create comprehensive visualizations for model analysis"""
    print("\n📈 Creating visualizations...")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # R² scores for each hormone and model
    model_names = list(results.keys())
    hormones = list(next(iter(results.values()))['metrics'].keys())

    # Plot 1: R² Comparison
    for i, hormone in enumerate(hormones):
        r2_scores = [results[model]['metrics'][hormone]['r2'] for model in model_names]
        bars = axes[0, 0].bar(np.arange(len(model_names)) + i * 0.2, r2_scores, width=0.2, label=hormone)

        # Add value labels
        for bar, score in zip(bars, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    axes[0, 0].set_xticks(np.arange(len(model_names)) + 0.2)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Performance by Hormone (R²)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Feature Importance Comparison (Top 10)
    axes[0, 1].axis('off')  # We'll use this for the combined importance plot later

    # Plot 3: Actual vs Predicted for best model
    best_model_name = max(results.keys(),
                          key=lambda x: np.mean([m['r2'] for m in results[x]['metrics'].values()]))
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)

    for i, hormone in enumerate(hormones):
        axes[1, 0].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6, label=hormone, s=20)

    max_val = max(y_test.max().max(), y_pred.max())
    min_val = min(y_test.min().min(), y_pred.min())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cross-validation Performance
    cv_means = [results[model]['cv_mean'] for model in model_names]
    cv_stds = [results[model]['cv_std'] for model in model_names]

    bars = axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    axes[1, 1].set_ylabel('Cross-Validated R²')
    axes[1, 1].set_title('Model Stability (Cross-Validation)')
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detailed Feature Importance Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, (model_name, importance_df) in enumerate(importance_results.items()):
        top_features = importance_df.head(15)

        axes[idx].barh(range(len(top_features)), top_features['importance_normalized'])
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['feature'])
        axes[idx].set_xlabel('Feature Importance (%)')
        axes[idx].set_title(f'{model_name} - Top 15 Features')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_comprehensive_report(results, importance_results, analysis_data, predictors):
    """Generate comprehensive analysis report"""
    print("\n📋 Generating comprehensive report...")

    report_content = []
    report_content.append("=" * 80)
    report_content.append("RANDOM FOREST & XGBOOST ANALYSIS REPORT")
    report_content.append("=" * 80)
    report_content.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")

    # Executive Summary
    report_content.append("1. EXECUTIVE SUMMARY")
    report_content.append("-" * 40)
    report_content.append(f"Total observations: {len(analysis_data)}")
    report_content.append(f"Number of predictors: {len(predictors)}")
    report_content.append(f"Target hormones: {list(next(iter(results.values()))['metrics'].keys())}")
    report_content.append("")

    # Model Performance
    report_content.append("2. MODEL PERFORMANCE")
    report_content.append("-" * 40)

    best_model = None
    best_score = -np.inf

    for model_name, result in results.items():
        report_content.append(f"{model_name}:")
        report_content.append(f"  Cross-Validation R²: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")

        for hormone, metrics in result['metrics'].items():
            report_content.append(f"  {hormone.upper():<10} - R²: {metrics['r2']:.3f}, "
                                  f"MSE: {metrics['mse']:.3f}, MAE: {metrics['mae']:.3f}")

        report_content.append("")

        # Track best model
        avg_r2 = np.mean([m['r2'] for m in result['metrics'].values()])
        if avg_r2 > best_score:
            best_score = avg_r2
            best_model = model_name

    report_content.append(f"Best Performing Model: {best_model} (Average R²: {best_score:.3f})")
    report_content.append("")

    # Feature Importance Summary
    report_content.append("3. FEATURE IMPORTANCE SUMMARY")
    report_content.append("-" * 40)
    report_content.append("Top 10 Most Important Features (Consensus):")

    # Get consensus top features (average across models)
    consensus_importance = {}
    for model_name, importance_df in importance_results.items():
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance_normalized']
            if feature not in consensus_importance:
                consensus_importance[feature] = []
            consensus_importance[feature].append(importance)

    # Calculate average importance
    consensus_df = pd.DataFrame([
        {'feature': feature, 'average_importance': np.mean(importances)}
        for feature, importances in consensus_importance.items()
    ]).sort_values('average_importance', ascending=False)

    for i, (_, row) in enumerate(consensus_df.head(10).iterrows()):
        report_content.append(f"  {i + 1:2d}. {row['feature']:<25} : {row['average_importance']:.2f}%")

    report_content.append("")

    # Save report
    report_text = "\n".join(report_content)
    with open(OUTPUT_DIR / 'tree_models_analysis_report.txt', 'w') as f:
        f.write(report_text)

    # Save feature importance data
    for model_name, importance_df in importance_results.items():
        importance_df.to_csv(OUTPUT_DIR / f'{model_name.replace(" ", "_").lower()}_importance.csv', index=False)

    consensus_df.to_csv(OUTPUT_DIR / 'consensus_feature_importance.csv', index=False)

    return report_text


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("RANDOM FOREST & XGBOOST ANALYSIS FOR HORMONE PREDICTION")
    print("=" * 80)

    try:
        # Step 1: Load and preprocess data
        merged_data = load_and_preprocess_data()
        merged_data = convert_symptoms_to_numeric(merged_data)
        merged_data = create_interaction_features(merged_data)

        print(f"📊 Final merged data shape: {merged_data.shape}")

        # Step 2: Select features for analysis
        analysis_data, predictors, targets = select_features_for_analysis(merged_data)

        # Step 3: Prepare X and y
        X = analysis_data[predictors].copy()
        y = analysis_data[targets].copy()

        print(f"🎯 Analysis data - X: {X.shape}, y: {y.shape}")

        # Step 4: Train models
        results, X_test, y_test = train_tree_models(X, y)

        if not results:
            print("❌ No models were successfully trained")
            return

        # Step 5: Analyze feature importance
        importance_results = analyze_feature_importance(results, predictors)

        # Step 6: Create visualizations
        create_comprehensive_visualizations(results, importance_results, X_test, y_test)

        # Step 7: Generate report
        report = generate_comprehensive_report(results, importance_results, analysis_data, predictors)

        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"📈 Data: {X.shape[0]} observations, {X.shape[1]} features")

        best_model = max(results.keys(),
                         key=lambda x: np.mean([m['r2'] for m in results[x]['metrics'].values()]))

        print(f"🏆 Best model: {best_model}")
        for hormone, metrics in results[best_model]['metrics'].items():
            print(f"  {hormone}: R² = {metrics['r2']:.3f}")

        print(f"\n🔑 Top 3 features:")
        consensus_importance = {}
        for model_name, importance_df in importance_results.items():
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance_normalized']
                if feature not in consensus_importance:
                    consensus_importance[feature] = []
                consensus_importance[feature].append(importance)

        consensus_df = pd.DataFrame([
            {'feature': feature, 'average_importance': np.mean(importances)}
            for feature, importances in consensus_importance.items()
        ]).sort_values('average_importance', ascending=False)

        for i, (_, row) in enumerate(consensus_df.head(3).iterrows()):
            print(f"  {i + 1}. {row['feature']} ({row['average_importance']:.1f}%)")

        print(f"\n📁 Report saved to: {OUTPUT_DIR}/")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()