# model_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.stats as stats
import warnings


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from sklearn.model_selection import KFold


warnings.filterwarnings('ignore')


def comprehensive_model_evaluation(df, best_variables, targets=['lh', 'estrogen', 'pdg']):
    """
    Comprehensive evaluation of XGBoost model with proper train/test split
    and multiple evaluation metrics
    """
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 50)

    # Check which variables actually exist in the dataframe
    available_variables = [var for var in best_variables if var in df.columns]
    print(f"Using {len(available_variables)} available variables")

    # Prepare data
    X = df[available_variables].copy()
    y = df[targets].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.describe()}")

    if len(X) < 100:
        print("⚠️ Not enough data for proper evaluation")
        return None

    # Split into train and test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"\n📊 Data Split:")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    # Impute missing values in features
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train XGBoost model
    print("\n🚀 Training XGBoost model...")
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X_train_imputed, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_imputed)
    y_test_pred = model.predict(X_test_imputed)

    # Convert to DataFrame for easier handling
    y_train_pred_df = pd.DataFrame(y_train_pred, columns=targets, index=y_train.index)
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=targets, index=y_test.index)

    # Calculate comprehensive metrics
    results = calculate_comprehensive_metrics(y_train, y_train_pred, y_test, y_test_pred, targets)

    # Create visualizations
    create_evaluation_visualizations(y_train, y_train_pred_df, y_test, y_test_pred_df, targets)

    # Feature importance analysis
    analyze_feature_importance(model, available_variables, targets)

    # Residual analysis
    analyze_residuals(y_test, y_test_pred_df, targets)

    return results, model, (X_train, X_test, y_train, y_test)


def calculate_comprehensive_metrics(y_train, y_train_pred, y_test, y_test_pred, targets):
    """Calculate multiple evaluation metrics"""
    print("\n📈 COMPREHENSIVE METRICS")
    print("=" * 40)

    results = {}

    for i, target in enumerate(targets):
        print(f"\n🎯 Target: {target.upper()}")
        print("-" * 20)

        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
        train_mae = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
        train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])

        # Testing metrics
        test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
        test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

        # Calculate mean absolute percentage error (MAPE)
        train_mape = np.mean(np.abs((y_train.iloc[:, i] - y_train_pred[:, i]) / y_train.iloc[:, i])) * 100
        test_mape = np.mean(np.abs((y_test.iloc[:, i] - y_test_pred[:, i]) / y_test.iloc[:, i])) * 100

        results[target] = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_mape': test_mape
        }

        print(f"TRAINING:")
        print(f"  RMSE:  {train_rmse:.4f}")
        print(f"  MAE:   {train_mae:.4f}")
        print(f"  R²:    {train_r2:.4f}")
        print(f"  MAPE:  {train_mape:.2f}%")

        print(f"TESTING:")
        print(f"  RMSE:  {test_rmse:.4f}")
        print(f"  MAE:   {test_mae:.4f}")
        print(f"  R²:    {test_r2:.4f}")
        print(f"  MAPE:  {test_mape:.2f}%")

        # Overfitting indicator
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.1:
            print(f"  ⚠️  Potential overfitting (R² gap: {r2_gap:.3f})")
        else:
            print(f"  ✅ Good generalization (R² gap: {r2_gap:.3f})")

    return results


def create_evaluation_visualizations(y_train, y_train_pred, y_test, y_test_pred, targets):
    """Create comprehensive visualizations"""
    print("\n📊 CREATING VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Model Performance Evaluation', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, target in enumerate(targets):
        color = colors[i % len(colors)]

        # Plot 1: Actual vs Predicted (Training)
        ax1 = axes[0, i]
        ax1.scatter(y_train.iloc[:, i], y_train_pred.iloc[:, i], alpha=0.6, color=color, label='Training')
        ax1.scatter(y_test.iloc[:, i], y_test_pred.iloc[:, i], alpha=0.6, color='red', label='Testing')

        # Perfect prediction line
        min_val = min(y_train.iloc[:, i].min(), y_test.iloc[:, i].min())
        max_val = max(y_train.iloc[:, i].max(), y_test.iloc[:, i].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)

        ax1.set_xlabel(f'Actual {target.upper()}')
        ax1.set_ylabel(f'Predicted {target.upper()}')
        ax1.set_title(f'Actual vs Predicted - {target.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals (Testing)
        ax2 = axes[1, i]
        residuals = y_test.iloc[:, i] - y_test_pred.iloc[:, i]
        ax2.scatter(y_test_pred.iloc[:, i], residuals, alpha=0.6, color=color)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel(f'Predicted {target.upper()}')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'Residual Plot - {target.upper()}')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create performance comparison plot
    plot_performance_comparison(y_test, y_test_pred, targets)


def plot_performance_comparison(y_test, y_test_pred, targets):
    """Plot side-by-side comparison of actual vs predicted values"""
    fig, axes = plt.subplots(1, len(targets), figsize=(15, 5))

    for i, target in enumerate(targets):
        # Sort for better visualization
        idx_sorted = y_test.iloc[:, i].argsort()
        y_test_sorted = y_test.iloc[:, i].iloc[idx_sorted]
        y_pred_sorted = y_test_pred.iloc[:, i].iloc[idx_sorted]

        axes[i].plot(range(len(y_test_sorted)), y_test_sorted.values,
                     label='Actual', color='blue', alpha=0.7, linewidth=2)
        axes[i].plot(range(len(y_pred_sorted)), y_pred_sorted.values,
                     label='Predicted', color='red', alpha=0.7, linewidth=2)

        axes[i].set_xlabel('Sample Index (sorted)')
        axes[i].set_ylabel(f'{target.upper()} Value')
        axes[i].set_title(f'Actual vs Predicted - {target.upper()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('actual_vs_predicted_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_feature_importance(model, feature_names, targets):
    """Analyze and visualize feature importance"""
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")

    # Create feature importance DataFrame
    importance_data = []

    for i, estimator in enumerate(model.estimators_):
        importance = estimator.feature_importances_
        for j, (feature, imp) in enumerate(zip(feature_names, importance)):
            importance_data.append({
                'hormone': targets[i],
                'variable': feature,
                'importance': imp
            })

    importance_df = pd.DataFrame(importance_data)

    # Calculate average importance
    avg_importance = importance_df.groupby('variable')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)

    print("\n📊 Top 15 Most Important Features:")
    print(avg_importance.head(15).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = avg_importance.head(15)

    plt.barh(range(len(top_features)), top_features['importance'][::-1])
    plt.yticks(range(len(top_features)), top_features['variable'][::-1])
    plt.xlabel('Feature Importance Score')
    plt.title('Top 15 Most Important Features (Average across all targets)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_residuals(y_test, y_test_pred, targets):
    """Analyze residuals for model diagnostics"""
    print("\n📊 RESIDUAL ANALYSIS")

    fig, axes = plt.subplots(1, len(targets), figsize=(15, 5))

    for i, target in enumerate(targets):
        residuals = y_test.iloc[:, i] - y_test_pred.iloc[:, i]

        # Normality test
        _, p_value = stats.normaltest(residuals.dropna())

        axes[i].hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[i].set_xlabel('Residuals')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Residual Distribution - {target.upper()}\n'
                          f'Normality p-value: {p_value:.4f}')
        axes[i].grid(True, alpha=0.3)

        print(f"{target.upper()} - Residual stats:")
        print(f"  Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
        print(f"  Normality test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  ⚠️  Residuals may not be normally distributed")
        else:
            print("  ✅ Residuals appear normally distributed")

    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def cross_validation_evaluation(df, best_variables, targets=['lh', 'estrogen', 'pdg'], k=5):
    """Perform k-fold cross validation for more robust evaluation"""
    print(f"\n🔄 PERFORMING {k}-FOLD CROSS VALIDATION")

    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error

    # Prepare data
    X = df[best_variables].copy()
    y = df[targets].copy()

    # Remove rows where targets are missing
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    cv_results = {target: {'r2_scores': [], 'rmse_scores': []} for target in targets}

    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"  Fold {fold}/{k}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Train model
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        )
        model.fit(X_train_imputed, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_imputed)

        for i, target in enumerate(targets):
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))

            cv_results[target]['r2_scores'].append(r2)
            cv_results[target]['rmse_scores'].append(rmse)

        fold += 1

    # Print cross-validation results
    print("\n📊 CROSS-VALIDATION RESULTS:")
    print("=" * 40)
    for target in targets:
        r2_scores = cv_results[target]['r2_scores']
        rmse_scores = cv_results[target]['rmse_scores']

        print(f"\n🎯 {target.upper()}:")
        print(f"  R²:  {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
        print(f"  RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")

    return cv_results


# Usage example (add this to your main script):
def evaluate_model_with_current_data():
    """Run comprehensive evaluation with your current data"""

    # Assuming you have your merged data and best_variables from the main script
    # You'll need to modify this based on your current data structure

    # Load your current dataset
    # current_data = pd.read_csv('your_merged_data.csv')  # Or use your existing variable

    # If you're running this within your main script, you can use:
    # results, model, data_splits = comprehensive_model_evaluation(current_data, best_variables)

    # For cross-validation (more robust):
    # cv_results = cross_validation_evaluation(current_data, best_variables)

    pass


# If running standalone, you'll need to load your data first
if __name__ == "__main__":
    print("Model Evaluation Script")
    print("Make sure to load your data first or run this within your main script")