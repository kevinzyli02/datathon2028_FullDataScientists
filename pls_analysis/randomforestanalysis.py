# random_forest_xgboost_analysis.py
"""
RANDOM FOREST & XGBOOST ANALYSIS FOR HORMONE PREDICTION
Captures non-linear relationships and provides robust feature importance
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('tree_models_report')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess data for tree-based models"""
    print("📂 Loading and preprocessing data...")

    # Load core datasets
    hormones = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(DATA_DIR / 'glucose.csv')
    sleep = pd.read_csv(DATA_DIR / 'sleep.csv')

    # Process glucose data
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std', 'min', 'max']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max']

    # Merge core datasets
    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    merged_data = pd.merge(merged_data, sleep,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')

    print(f"Core data shape: {merged_data.shape}")

    # Load and merge optional datasets
    optional_files = ['active_minutes.csv', 'resting_heart_rate.csv', 'stress_score.csv']
    for file in optional_files:
        file_path = DATA_DIR / file
        if file_path.exists():
            df = pd.read_csv(file_path)
            if file == 'active_minutes.csv':
                merged_data = pd.merge(merged_data, df, on=['id', 'day_in_study'], how='left')
            elif file == 'resting_heart_rate.csv':
                merged_data = pd.merge(merged_data, df.rename(columns={'value': 'resting_hr'}),
                                       on=['id', 'day_in_study'], how='left')
            elif file == 'stress_score.csv':
                merged_data = pd.merge(merged_data, df, on=['id', 'day_in_study'], how='left')
            print(f"Added {file}: {merged_data.shape}")

    return merged_data


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
            df[col] = df[col].map(symptom_mapping)

    return df


def create_interaction_features(df):
    """Create biologically relevant interaction terms"""
    print("🔄 Creating interaction features...")

    # Activity interactions
    if all(col in df.columns for col in ['lightly', 'moderately', 'very']):
        df['total_activity'] = df['lightly'] + df['moderately'] + df['very']
        df['activity_variability'] = df[['lightly', 'moderately', 'very']].std(axis=1)
        df['intense_ratio'] = df['very'] / (df['lightly'] + 1)

    # Stress-sleep interactions
    if all(col in df.columns for col in ['stress_score', 'efficiency']):
        df['stress_sleep_ratio'] = df['stress_score'] / (df['efficiency'] + 1)

    # Glucose variability
    if 'glucose_std' in df.columns:
        df['glucose_cv'] = df['glucose_std'] / (df['glucose_mean'] + 1)

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
        # Interaction terms (will be created)
        'total_activity', 'activity_variability', 'intense_ratio',
        'stress_sleep_ratio', 'glucose_cv'
    ]

    # Filter to available columns
    available_predictors = [col for col in all_predictors if col in merged_data.columns]
    targets = ['lh', 'estrogen', 'pdg']

    print(f"Available predictors: {len(available_predictors)}")
    print(f"Targets: {targets}")

    # Select data for analysis
    analysis_data = merged_data[available_predictors + targets].copy()

    # Remove columns with too many missing values (>90% missing)
    missing_threshold = len(analysis_data) * 0.1  # Keep if at least 10% data
    columns_to_keep = []
    for col in available_predictors:
        if analysis_data[col].notna().sum() >= missing_threshold:
            columns_to_keep.append(col)

    print(f"Predictors after missing value filter: {len(columns_to_keep)}")

    return analysis_data, columns_to_keep, targets


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_tree_models(X, y, test_size=0.2, random_state=42):
    """Train Random Forest and XGBoost models with comprehensive evaluation"""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

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
        cv_scores = cross_val_score(model, X_train_imputed, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)

        results[model_name] = {
            'model': model,
            'metrics': hormone_metrics,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'feature_names': X.columns.tolist(),
            'imputer': imputer
        }

        print(f"✅ {model_name} - CV R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

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
        axes[1, 0].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6, label=hormone)

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

    # 3. Combined Feature Importance
    plt.figure(figsize=(12, 8))

    # Get top 10 features from both models and combine
    all_important_features = set()
    for importance_df in importance_results.values():
        all_important_features.update(importance_df.head(10)['feature'].tolist())

    # Create combined importance dataframe
    combined_data = []
    for feature in all_important_features:
        row = {'feature': feature}
        for model_name, importance_df in importance_results.items():
            model_importance = importance_df[importance_df['feature'] == feature]['importance_normalized']
            row[model_name] = model_importance.values[0] if len(model_importance) > 0 else 0
        combined_data.append(row)

    combined_df = pd.DataFrame(combined_data)
    combined_df['average_importance'] = combined_df[list(importance_results.keys())].mean(axis=1)
    combined_df = combined_df.sort_values('average_importance', ascending=True)

    # Plot combined importance
    y_pos = np.arange(len(combined_df))
    height = 0.35

    for i, model_name in enumerate(importance_results.keys()):
        plt.barh(y_pos + i * height, combined_df[model_name], height,
                 label=model_name, alpha=0.8)

    plt.yticks(y_pos + height / 2, combined_df['feature'])
    plt.xlabel('Feature Importance (%)')
    plt.title('Combined Feature Importance Across Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_feature_importance.png', dpi=300, bbox_inches='tight')
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

    # Biological Insights
    report_content.append("4. BIOLOGICAL INSIGHTS")
    report_content.append("-" * 40)

    top_features = consensus_df.head(5)['feature'].tolist()
    report_content.append("Most Important Predictors:")
    for feature in top_features:
        if 'stress' in feature.lower():
            report_content.append(f"  • {feature}: Stress management appears crucial for hormonal balance")
        elif any(activity in feature for activity in ['lightly', 'moderately', 'very', 'activity']):
            report_content.append(f"  • {feature}: Physical activity levels significantly influence hormones")
        elif 'sleep' in feature.lower() or 'efficiency' in feature.lower():
            report_content.append(f"  • {feature}: Sleep quality is a key hormonal regulator")
        elif 'glucose' in feature.lower():
            report_content.append(f"  • {feature}: Glucose metabolism shows strong relationship with hormones")
        else:
            report_content.append(f"  • {feature}: Important predictor of hormonal levels")

    report_content.append("")

    # Recommendations
    report_content.append("5. RECOMMENDATIONS")
    report_content.append("-" * 40)
    report_content.append("Based on Feature Importance:")
    report_content.append("• Focus on stress reduction and management strategies")
    report_content.append("• Maintain consistent physical activity across intensity levels")
    report_content.append("• Prioritize sleep quality and duration")
    report_content.append("• Monitor glucose levels and maintain metabolic health")
    report_content.append("• The top 5 features should be primary monitoring targets")

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

    # Step 1: Load and preprocess data
    merged_data = load_and_preprocess_data()
    merged_data = convert_symptoms_to_numeric(merged_data)
    merged_data = create_interaction_features(merged_data)

    # Step 2: Select features for analysis
    analysis_data, predictors, targets = select_features_for_analysis(merged_data)

    # Step 3: Prepare X and y
    X = analysis_data[predictors].copy()
    y = analysis_data[targets].copy()

    # Remove rows where all targets are missing
    valid_indices = y.notna().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    print(f"Final analysis data: X={X.shape}, y={y.shape}")

    # Step 4: Train models
    results, X_test, y_test = train_tree_models(X, y)

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
    print(f"Data: {X.shape[0]} observations, {X.shape[1]} features")
    print(f"Best model performance:")

    best_model = max(results.keys(),
                     key=lambda x: np.mean([m['r2'] for m in results[x]['metrics'].values()]))

    for hormone, metrics in results[best_model]['metrics'].items():
        print(f"  {hormone}: R² = {metrics['r2']:.3f}")

    print(f"\nTop 3 features:")
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

    print(f"\nReport saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()