# optimized_xgboost_model.py
"""
OPTIMIZED XGBOOST MODEL USING BEST VARIABLES
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
EXCLUSION_FILE = Path('/Users/kevin/Downloads/McPhases SelfReport Regular Cycles Labelled.csv')
MODEL_DIR = Path('optimized_xgboost_model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Best variables from your analysis
BEST_VARIABLES = [
    'stress_score',  # 21.87% importance
    'lightly',  # 14.23% importance
    'total_activity',  # 12.88% importance
    'very',  # 12.86% importance
    'moderately',  # 8.40% importance
    'minutesawake',  # 7.96% importance
    'stress_sleep_ratio',  # 7.82% importance
    'intense_ratio',  # 7.73% importance
    'resting_hr',  # 2.29% importance
    'minutesasleep'  # 2.16% importance
]

TARGETS = ['lh', 'estrogen', 'pdg']


# =============================================================================
# DATA PREPARATION WITH BEST VARIABLES
# =============================================================================

def prepare_optimized_data():
    """Prepare data using only the best variables"""
    print("📊 Preparing data with optimized feature set...")

    # Load core datasets
    hormones = pd.read_csv(DATA_DIR / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(DATA_DIR / 'glucose.csv')
    sleep = pd.read_csv(DATA_DIR / 'sleep.csv')
    active_minutes = pd.read_csv(DATA_DIR / 'active_minutes.csv')
    resting_hr = pd.read_csv(DATA_DIR / 'resting_heart_rate.csv')
    stress = pd.read_csv(DATA_DIR / 'stress_score.csv')

    # Process glucose data
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std']

    # Start with hormones as base
    merged_data = hormones.copy()

    # Merge glucose data
    merged_data = pd.merge(merged_data, glucose_daily, on=['id', 'day_in_study'], how='left')

    # Merge sleep data
    sleep_renamed = sleep.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
    sleep_cols = ['id', 'day_in_study', 'minutesasleep', 'minutesawake']
    merged_data = pd.merge(merged_data, sleep_renamed[sleep_cols], on=['id', 'day_in_study'], how='left')

    # Merge active minutes
    merged_data = pd.merge(merged_data, active_minutes, on=['id', 'day_in_study'], how='left')

    # Merge resting heart rate
    resting_hr_renamed = resting_hr.rename(columns={'value': 'resting_hr'})
    merged_data = pd.merge(merged_data, resting_hr_renamed[['id', 'day_in_study', 'resting_hr']],
                           on=['id', 'day_in_study'], how='left')

    # Merge stress score
    merged_data = pd.merge(merged_data, stress[['id', 'day_in_study', 'stress_score']],
                           on=['id', 'day_in_study'], how='left')

    print(f"📈 Raw merged data: {merged_data.shape}")

    # Create derived features
    merged_data = create_derived_features(merged_data)

    # Filter to only include our best variables + targets
    available_vars = [var for var in BEST_VARIABLES if var in merged_data.columns]
    print(f"✅ Using {len(available_vars)} best variables: {available_vars}")

    analysis_data = merged_data[available_vars + TARGETS].copy()

    # Remove rows where all targets are missing
    analysis_data = analysis_data.dropna(subset=TARGETS, how='all')
    print(f"🎯 Final analysis data: {analysis_data.shape}")

    return analysis_data, available_vars


def create_derived_features(df):
    """Create the derived features from our best variables list"""

    # Create interaction features that were important
    if all(col in df.columns for col in ['lightly', 'moderately', 'very']):
        df['total_activity'] = df['lightly'] + df['moderately'] + df['very']
        df['intense_ratio'] = df['very'] / (df['lightly'] + 1)

    if all(col in df.columns for col in ['stress_score', 'minutesasleep']):
        df['stress_sleep_ratio'] = df['stress_score'] / (df['minutesasleep'] + 1)

    return df


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def optimize_xgboost_hyperparameters(X, y):
    """Find the best hyperparameters for XGBoost"""
    print("\n🎯 Optimizing XGBoost hyperparameters...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Parameter grid for optimization
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 6, 9],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Use RandomizedSearchCV for faster optimization
    xgb = MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1))

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_imputed, y_train)

    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation score: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_, imputer, grid_search.best_params_


# =============================================================================
# MODEL TRAINING WITH OPTIMAL PARAMETERS
# =============================================================================

def train_optimized_xgboost(X, y, best_params=None):
    """Train XGBoost with optimal parameters"""
    print("\n🏋️ Training optimized XGBoost model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Use best parameters if provided, otherwise use sensible defaults
    if best_params:
        model = MultiOutputRegressor(
            XGBRegressor(
                **best_params,
                random_state=42,
                n_jobs=-1
            )
        )
    else:
        # Sensible defaults based on typical best practices
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
        )

    # Train model
    model.fit(X_train_imputed, y_train)

    # Predictions
    y_pred = model.predict(X_test_imputed)

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"✅ Model trained successfully!")
    print(f"📊 Cross-validation R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    return model, imputer, metrics, X_test_imputed, y_test, y_pred


def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    metrics = {}

    for i, hormone in enumerate(y_true.columns):
        r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)

        # Calculate mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((y_true.iloc[:, i] - y_pred[:, i]) / (y_true.iloc[:, i] + 1e-8))) * 100

        metrics[hormone] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    return metrics


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_xgboost_importance(model, feature_names):
    """Analyze feature importance from the trained XGBoost model"""
    print("\n🔍 Analyzing feature importance...")

    # Get feature importance from all estimators (one per target)
    importance_data = []

    for i, estimator in enumerate(model.estimators_):
        importance = estimator.feature_importances_
        for j, (feature, imp) in enumerate(zip(feature_names, importance)):
            importance_data.append({
                'hormone': TARGETS[i],
                'feature': feature,
                'importance': imp
            })

    importance_df = pd.DataFrame(importance_data)

    # Calculate average importance across all hormones
    avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)
    avg_importance['importance_pct'] = avg_importance['importance'] / avg_importance['importance'].sum() * 100

    print("📊 Average Feature Importance:")
    for i, (_, row) in enumerate(avg_importance.iterrows()):
        print(f"   {i + 1:2d}. {row['feature']:<20} : {row['importance_pct']:.2f}%")

    return importance_df, avg_importance


# =============================================================================
# MODEL EVALUATION & VISUALIZATION
# =============================================================================

def create_model_evaluation_plots(y_test, y_pred, metrics, importance_df):
    """Create comprehensive evaluation visualizations"""
    print("\n📈 Creating evaluation plots...")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: R² scores for each hormone
    hormones = list(metrics.keys())
    r2_scores = [metrics[h]['r2'] for h in hormones]

    bars = axes[0, 0].bar(hormones, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Performance by Hormone (R²)')
    axes[0, 0].grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, r2_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

    # Plot 2: Feature Importance
    top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10)
    axes[0, 1].barh(range(len(top_features)), top_features.values)
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features.index)
    axes[0, 1].set_xlabel('Feature Importance')
    axes[0, 1].set_title('Top 10 Feature Importance')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].invert_yaxis()

    # Plot 3: Actual vs Predicted scatter plots
    for i, hormone in enumerate(hormones):
        axes[1, 0].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6, label=hormone, s=30)

    max_val = max(y_test.max().max(), y_pred.max())
    min_val = min(y_test.min().min(), y_pred.min())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error metrics comparison
    error_metrics = ['mae', 'rmse', 'mape']
    metric_data = []
    for hormone in hormones:
        for metric in error_metrics:
            metric_data.append({
                'hormone': hormone,
                'metric': metric.upper(),
                'value': metrics[hormone][metric]
            })

    error_df = pd.DataFrame(metric_data)

    # Normalize MAPE for better visualization
    mape_data = error_df[error_df['metric'] == 'MAPE']['value']
    if not mape_data.empty:
        error_df.loc[error_df['metric'] == 'MAPE', 'value'] = mape_data / mape_data.max()

    sns.barplot(data=error_df, x='hormone', y='value', hue='metric', ax=axes[1, 1])
    axes[1, 1].set_title('Error Metrics by Hormone')
    axes[1, 1].set_ylabel('Normalized Error')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'xgboost_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detailed feature importance by hormone
    plt.figure(figsize=(12, 8))

    # Pivot for heatmap
    importance_pivot = importance_df.pivot(index='feature', columns='hormone', values='importance')
    importance_pivot = importance_pivot.loc[top_features.index]  # Use top features

    sns.heatmap(importance_pivot, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance by Hormone')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'feature_importance_by_hormone.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MODEL SAVING & DEPLOYMENT
# =============================================================================

def save_model_and_artifacts(model, imputer, feature_names, metrics, importance_df):
    """Save the trained model and all artifacts"""
    print("\n💾 Saving model and artifacts...")

    # Save the model
    joblib.dump(model, MODEL_DIR / 'optimized_xgboost_model.pkl')

    # Save the imputer
    joblib.dump(imputer, MODEL_DIR / 'imputer.pkl')

    # Save feature names
    with open(MODEL_DIR / 'feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(MODEL_DIR / 'model_metrics.csv')

    # Save feature importance
    importance_df.to_csv(MODEL_DIR / 'feature_importance.csv', index=False)

    # Create deployment info
    deployment_info = {
        'model_type': 'XGBoost MultiOutput Regressor',
        'features_used': feature_names,
        'targets': TARGETS,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance': {hormone: {k: v for k, v in metrics[hormone].items() if k == 'r2'}
                        for hormone in metrics.keys()}
    }

    import json
    with open(MODEL_DIR / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print(f"✅ Model and artifacts saved to {MODEL_DIR}/")


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_hormones(model, imputer, new_data):
    """Make predictions on new data using the trained model"""

    # Ensure we have the right features
    required_features = [f for f in BEST_VARIABLES if f in new_data.columns]
    missing_features = set(BEST_VARIABLES) - set(required_features)

    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        # You might want to handle this differently in production

    # Select and order features correctly
    X_new = new_data[required_features].copy()

    # Create any missing derived features
    if 'total_activity' not in X_new.columns and all(f in X_new.columns for f in ['lightly', 'moderately', 'very']):
        X_new['total_activity'] = X_new['lightly'] + X_new['moderately'] + X_new['very']

    if 'intense_ratio' not in X_new.columns and all(f in X_new.columns for f in ['very', 'lightly']):
        X_new['intense_ratio'] = X_new['very'] / (X_new['lightly'] + 1)

    if 'stress_sleep_ratio' not in X_new.columns and all(f in X_new.columns for f in ['stress_score', 'minutesasleep']):
        X_new['stress_sleep_ratio'] = X_new['stress_score'] / (X_new['minutesasleep'] + 1)

    # Impute missing values
    X_new_imputed = imputer.transform(X_new)

    # Make predictions
    predictions = model.predict(X_new_imputed)

    # Create results dataframe
    results = pd.DataFrame(predictions, columns=TARGETS, index=new_data.index)

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("OPTIMIZED XGBOOST HORMONE PREDICTION MODEL")
    print("=" * 80)

    try:
        # Step 1: Prepare data with best variables
        analysis_data, feature_names = prepare_optimized_data()

        # Prepare X and y
        X = analysis_data[feature_names].copy()
        y = analysis_data[TARGETS].copy()

        # Remove rows where all targets are missing
        valid_indices = y.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

        print(f"🎯 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Step 2: Hyperparameter optimization (optional - can skip for speed)
        use_hyperparameter_optimization = False  # Set to True if you want to optimize

        if use_hyperparameter_optimization:
            best_model, imputer, best_params = optimize_xgboost_hyperparameters(X, y)
        else:
            best_params = None

        # Step 3: Train optimized model
        model, imputer, metrics, X_test, y_test, y_pred = train_optimized_xgboost(X, y, best_params)

        # Step 4: Analyze feature importance
        importance_df, avg_importance = analyze_xgboost_importance(model, feature_names)

        # Step 5: Create evaluation visualizations
        create_model_evaluation_plots(y_test, y_pred, metrics, importance_df)

        # Step 6: Save model and artifacts
        save_model_and_artifacts(model, imputer, feature_names, metrics, importance_df)

        # Step 7: Print final summary
        print("\n" + "=" * 80)
        print("MODEL TRAINING COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"📊 Model Performance (R² Scores):")
        for hormone, metric in metrics.items():
            print(f"   {hormone.upper():<10}: {metric['r2']:.3f}")

        print(f"\n🔑 Top 5 Most Important Features:")
        for i, (_, row) in enumerate(avg_importance.head(5).iterrows()):
            print(f"   {i + 1}. {row['feature']:<20} ({row['importance_pct']:.1f}%)")

        print(f"\n💾 Model saved to: {MODEL_DIR}/")
        print("=" * 80)

        return model, imputer, feature_names

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    model, imputer, feature_names = main()