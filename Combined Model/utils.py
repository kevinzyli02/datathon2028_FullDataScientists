import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import config

def calculate_comprehensive_metrics(y_true, y_pred, model_name, target_name, n_features, n_patients):
    """Print and return comprehensive regression metrics."""
    print(f"\n📊 REGRESSION STATISTICS - {model_name} - {target_name}")
    print("=" * 50)
    print(f"   Features used: {n_features}")
    print(f"   Patients in set: {n_patients}")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    try:
        pearson_corr, _ = pearsonr(y_true, y_pred)
    except:
        pearson_corr = np.nan
    try:
        spearman_corr, _ = spearmanr(y_true, y_pred)
    except:
        spearman_corr = np.nan

    mean_true = np.mean(y_true)
    std_true = np.std(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    print("ERROR METRICS:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("CORRELATION METRICS:")
    print(f"  R2: {r2:.4f}")
    print(f"  Pearson: {pearson_corr:.4f}")
    print(f"  Spearman: {spearman_corr:.4f}")
    print("DATA STATISTICS:")
    print(f"  True Mean: {mean_true:.4f}, Std: {std_true:.4f}")

    return {
        'model': model_name,
        'target': target_name,
        'n_features': n_features,
        'n_patients': n_patients,
        'error_metrics': {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape},
        'correlation_metrics': {'r2': r2, 'pearson': pearson_corr, 'spearman': spearman_corr},
        'distribution_stats': {'true_mean': mean_true, 'true_std': std_true}
    }

def create_regression_diagnostics(y_true, y_pred, model_name, target_name, n_features, n_patients):
    """Create diagnostic plots and save to OUTPUT_DIR."""
    fig, axes = plt.subplots(2, 2, figsize=(15,12))
    fig.suptitle(f'Regression Diagnostics: {model_name} - {target_name}\nFeatures: {n_features}, Patients: {n_patients}', fontsize=16, fontweight='bold')

    # True vs Predicted
    axes[0,0].scatter(y_true, y_pred, alpha=0.6)
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0,0].set_xlabel('True')
    axes[0,0].set_ylabel('Predicted')
    axes[0,0].set_title('True vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[0,0].plot(y_true, p(y_true), "b-", alpha=0.8)

    # Residuals
    residuals = y_true - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6)
    axes[0,1].axhline(0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)

    # Distributions
    axes[1,0].hist(y_true, bins=30, alpha=0.7, label='True', density=True)
    axes[1,0].hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
    axes[1,0].set_xlabel('Value')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distribution Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Error distribution
    axes[1,1].hist(residuals, bins=30, alpha=0.7, color='orange')
    axes[1,1].axvline(0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = f'regression_{model_name}_{target_name}'.replace(' ', '_')
    plt.savefig(config.OUTPUT_DIR / f'{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_csv(results, feature_sets, stage='hormone'):
    """Save performance results to CSV."""
    import pandas as pd
    perf_data = []
    for target, target_results in results.items():
        for model_name, res in target_results.items():
            if res and 'test_metrics' in res:
                m = res['test_metrics']
                perf_data.append({
                    'target': target,
                    'model': model_name,
                    'n_patients': m['n_patients'],
                    'n_features': m['n_features'],
                    'r2': m['correlation_metrics']['r2'],
                    'pearson_r': m['correlation_metrics']['pearson'],
                    'spearman_r': m['correlation_metrics']['spearman'],
                    'mae': m['error_metrics']['mae'],
                    'mse': m['error_metrics']['mse'],
                    'rmse': m['error_metrics']['rmse'],
                    'mape': m['error_metrics']['mape'],
                    'training_time': res['training_time']
                })
    if perf_data:
        pd.DataFrame(perf_data).to_csv(config.OUTPUT_DIR / f'{stage}_performance_comparison.csv', index=False)
        print(f"✅ {stage} performance comparison saved as CSV")