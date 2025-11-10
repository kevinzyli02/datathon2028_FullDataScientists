# optimized_xgboost_model.py
"""
OPTIMIZED XGBOOST MODEL WITH TEST SET EVALUATION
Faster version with top variable selection
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data')
OUTPUT_DIR = Path('optimized_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Focus on key files for faster processing
KEY_FILES = [
    'hormones_and_selfreport.csv',
    'active_minutes.csv',
    'sleep.csv',
    'stress_score.csv',
    'steps.csv',
    'resting_heart_rate.csv',
    'demographic_vo2_max.csv',
    'height_and_weight.csv'
]

TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42


# =============================================================================
# OPTIMIZED DATA LOADING & MERGING
# =============================================================================

def load_key_datasets():
    """Load only key datasets for faster processing"""
    print("📂 Loading key datasets...")

    datasets = {}

    for file in KEY_FILES:
        file_path = DATA_DIR / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                datasets[file] = df
                print(f"   ✅ {file}: {df.shape}")
            except Exception as e:
                print(f"   ❌ {file}: Error - {e}")
        else:
            print(f"   ❌ {file}: Not found")

    return datasets


def create_optimized_dataset(datasets):
    """Create optimized dataset with key variables only"""
    print("\n🔄 Creating optimized dataset...")

    # Start with hormones as base
    merged_data = datasets['hormones_and_selfreport.csv'].copy()
    print(f"   Base hormones data: {merged_data.shape}")

    # Simple merge strategy for key files
    merge_config = {
        'active_minutes.csv': ['id', 'day_in_study'],
        'sleep.csv': ['id', 'sleep_start_day_in_study'],
        'stress_score.csv': ['id', 'day_in_study'],
        'steps.csv': ['id', 'day_in_study'],
        'resting_heart_rate.csv': ['id', 'day_in_study'],
        'demographic_vo2_max.csv': ['id'],
        'height_and_weight.csv': ['id']
    }

    for file, merge_cols in merge_config.items():
        if file in datasets:
            try:
                df = datasets[file]

                # For sleep data, rename the day column
                if file == 'sleep.csv' and 'sleep_start_day_in_study' in df.columns:
                    df = df.rename(columns={'sleep_start_day_in_study': 'day_in_study'})

                # Simple direct merge
                merged_data = pd.merge(merged_data, df, on=merge_cols, how='left',
                                       suffixes=('', f'_{file.replace(".csv", "")}'))
                print(f"   ✅ Merged {file}: {merged_data.shape}")

            except Exception as e:
                print(f"   ⚠️  Error merging {file}: {e}")

    return merged_data


# =============================================================================
# FAST PREPROCESSING
# =============================================================================

def fast_preprocessing(df):
    """Fast preprocessing focusing on key variables"""
    print("\n⚡ Fast preprocessing...")

    # Convert symptoms to numeric quickly
    symptom_mapping = {'Not at all': 0, 'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
    symptom_cols = [col for col in ['appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts',
                                    'fatigue', 'sleepissue', 'moodswing', 'stress', 'foodcravings'] if
                    col in df.columns]

    for col in symptom_cols:
        df[col] = df[col].map(symptom_mapping)

    # Select only numeric columns for modeling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = TARGETS + ['id', 'day_in_study', 'sleep_start_day_in_study']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"   Using {len(feature_cols)} numeric features")
    return df, feature_cols


# =============================================================================
# VARIABLE SELECTION & MODEL TRAINING
# =============================================================================

def select_top_variables(df, feature_cols, top_k=20):
    """Select top K variables based on correlation with targets"""
    print(f"\n🎯 Selecting top {top_k} variables...")

    # Calculate correlation with each target
    corr_results = []

    for feature in feature_cols:
        if df[feature].notna().sum() > len(df) * 0.1:  # At least 10% complete
            corr_values = []
            for target in TARGETS:
                if target in df.columns:
                    valid_idx = df[[feature, target]].dropna().index
                    if len(valid_idx) > 10:  # Enough data points
                        corr = df.loc[valid_idx, feature].corr(df.loc[valid_idx, target])
                        if not np.isnan(corr):
                            corr_values.append(abs(corr))

            if corr_values:
                avg_corr = np.mean(corr_values)
                corr_results.append({
                    'variable': feature,
                    'avg_abs_correlation': avg_corr,
                    'completeness': df[feature].notna().sum() / len(df)
                })

    # Sort by correlation strength
    corr_df = pd.DataFrame(corr_results)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('avg_abs_correlation', ascending=False)
        top_variables = corr_df.head(top_k)['variable'].tolist()

        print("📊 Top 10 variables by correlation:")
        for i, (_, row) in enumerate(corr_df.head(10).iterrows()):
            print(f"   {i + 1:2d}. {row['variable']:<30} : {row['avg_abs_correlation']:.3f}")

        return top_variables
    else:
        print("   ⚠️  No correlations calculated, using all features")
        return feature_cols[:top_k]


def train_test_evaluation(df, feature_cols, model_name="Full Feature Model"):
    """Train and evaluate model with train/test split"""
    print(f"\n🤖 Training {model_name}...")

    # Prepare data
    X = df[feature_cols].copy()
    y = df[TARGETS].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"   Data shape: {X.shape}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train model (lightweight settings for speed)
    start_time = time.time()
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=4,  # Reduced for speed
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    model.fit(X_train_imputed, y_train)
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train_imputed)
    y_test_pred = model.predict(X_test_imputed)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Individual target R² scores
    target_r2 = {}
    for i, target in enumerate(TARGETS):
        target_r2[target] = {
            'train': r2_score(y_train.iloc[:, i], y_train_pred[:, i]),
            'test': r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        }

    print(f"   ✅ Training completed in {training_time:.1f}s")
    print(f"   📊 R² Score - Train: {train_r2:.3f}, Test: {test_r2:.3f}")

    for target in TARGETS:
        print(f"      {target}: Train R² = {target_r2[target]['train']:.3f}, Test R² = {target_r2[target]['test']:.3f}")

    return {
        'model_name': model_name,
        'model': model,
        'imputer': imputer,
        'feature_cols': feature_cols,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'target_r2': target_r2,
        'training_time': training_time,
        'n_features': len(feature_cols)
    }


# =============================================================================
# COMPARATIVE MODEL ANALYSIS
# =============================================================================

def compare_models(df, feature_cols):
    """Compare models with different feature sets"""
    print("\n🔬 Comparing different feature sets...")

    results = []

    # Model 1: All features (baseline)
    if len(feature_cols) > 30:
        all_features_result = train_test_evaluation(df, feature_cols[:30], "Top 30 Features")
    else:
        all_features_result = train_test_evaluation(df, feature_cols, "All Features")
    results.append(all_features_result)

    # Model 2: Top 10 variables
    top_10_vars = select_top_variables(df, feature_cols, top_k=10)
    top_10_result = train_test_evaluation(df, top_10_vars, "Top 10 Features")
    results.append(top_10_result)

    # Model 3: Top 5 variables
    top_5_vars = select_top_variables(df, feature_cols, top_k=5)
    top_5_result = train_test_evaluation(df, top_5_vars, "Top 5 Features")
    results.append(top_5_result)

    return results


def create_comparison_plot(results):
    """Create comparison plot of different models"""
    plt.figure(figsize=(12, 8))

    # Model names and scores
    model_names = [result['model_name'] for result in results]
    train_scores = [result['train_r2'] for result in results]
    test_scores = [result['test_r2'] for result in results]
    n_features = [result['n_features'] for result in results]

    # Create subplots
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(model_names))
    width = 0.35

    plt.bar(x_pos - width / 2, train_scores, width, label='Train R²', alpha=0.7)
    plt.bar(x_pos + width / 2, test_scores, width, label='Test R²', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width / 2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width / 2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=8)

    # Feature count vs performance
    plt.subplot(2, 2, 2)
    plt.scatter(n_features, test_scores, s=100, alpha=0.7)
    for i, (n_feat, score, name) in enumerate(zip(n_features, test_scores, model_names)):
        plt.annotate(name, (n_feat, score), xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Number of Features')
    plt.ylabel('Test R² Score')
    plt.title('Features vs Performance')
    plt.grid(True, alpha=0.3)

    # Individual target performance
    plt.subplot(2, 2, 3)
    target_data = {target: [] for target in TARGETS}
    for result in results:
        for target in TARGETS:
            target_data[target].append(result['target_r2'][target]['test'])

    x_pos = np.arange(len(results))
    for i, target in enumerate(TARGETS):
        plt.bar(x_pos + i * 0.25, target_data[target], 0.25, label=target, alpha=0.7)

    plt.xlabel('Models')
    plt.ylabel('Test R² Score')
    plt.title('Performance by Target Variable')
    plt.xticks(x_pos + 0.25, [result['model_name'] for result in results], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Training time comparison
    plt.subplot(2, 2, 4)
    training_times = [result['training_time'] for result in results]
    plt.bar(model_names, training_times, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, time_val in enumerate(training_times):
        plt.text(i, time_val + 0.1, f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_final_model(results, df, feature_cols):
    """Save the best model and results"""
    print("\n💾 Saving best model...")

    # Find best model based on test R²
    best_result = max(results, key=lambda x: x['test_r2'])

    print(f"🎉 Best Model: {best_result['model_name']}")
    print(f"   Test R²: {best_result['test_r2']:.3f}")
    print(f"   Features: {best_result['n_features']}")

    # Save model info
    model_info = {
        'best_model_name': best_result['model_name'],
        'test_r2': best_result['test_r2'],
        'n_features': best_result['n_features'],
        'feature_names': best_result['feature_cols'],
        'training_time': best_result['training_time']
    }

    # Save to files
    pd.DataFrame([model_info]).to_csv(OUTPUT_DIR / 'best_model_info.csv', index=False)

    # Save feature importance if available
    if hasattr(best_result['model'].estimators_[0], 'feature_importances_'):
        importance_data = []
        for i, estimator in enumerate(best_result['model'].estimators_):
            for j, (feature, importance) in enumerate(zip(best_result['feature_cols'], estimator.feature_importances_)):
                importance_data.append({
                    'hormone': TARGETS[i],
                    'feature': feature,
                    'importance': importance
                })

        importance_df = pd.DataFrame(importance_data)
        importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    # Save comparison results
    results_df = pd.DataFrame([
        {
            'model_name': r['model_name'],
            'n_features': r['n_features'],
            'train_r2': r['train_r2'],
            'test_r2': r['test_r2'],
            'training_time': r['training_time']
        } for r in results
    ])
    results_df.to_csv(OUTPUT_DIR / 'model_comparison_results.csv', index=False)

    return best_result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("OPTIMIZED XGBOOST MODEL WITH TEST SET EVALUATION")
    print("=" * 80)

    start_time = time.time()

    try:
        # Step 1: Load key datasets only
        datasets = load_key_datasets()

        # Step 2: Create optimized dataset
        merged_data = create_optimized_dataset(datasets)
        print(f"\n📊 Final dataset: {merged_data.shape}")

        # Step 3: Fast preprocessing
        processed_data, all_features = fast_preprocessing(merged_data)

        # Step 4: Compare different feature sets
        results = compare_models(processed_data, all_features)

        # Step 5: Create visualizations
        create_comparison_plot(results)

        # Step 6: Save best model
        best_model = save_final_model(results, processed_data, all_features)

        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.1f} seconds")
        print(f"📁 Results saved to: {OUTPUT_DIR}/")
        print("=" * 80)

        return best_model, processed_data

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    best_model, data = main()