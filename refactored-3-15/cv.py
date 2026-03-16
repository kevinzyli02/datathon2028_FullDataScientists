import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def run_patient_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    models: Dict[str, Any],
    n_splits: int = 3,
    target_name: str = "",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run GroupKFold cross‑validation with preprocessing inside each fold.

    Args:
        X: Feature matrix.
        y: Target vector.
        patient_ids: Array of patient IDs (same length as y).
        models: Dict of model instances (will be cloned inside each fold).
        n_splits: Number of folds.
        target_name: Name of target for logging.
        random_state: Random seed.

    Returns:
        DataFrame with fold‑level metrics.
    """
    gkf = GroupKFold(n_splits=n_splits)
    records = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=patient_ids), 1):
        logger.debug(f"   Fold {fold}/{n_splits}")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Preprocess: fit on training fold only
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_imp = imputer.fit_transform(X_tr)
        X_val_imp = imputer.transform(X_val)
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        for name, model in models.items():
            # Special handling for MixedEffects
            if name == "MixedEffects":
                # For simplicity, we assume model has a .fit(X,y,patient_ids) method
                model.fit(X_tr_scaled, y_tr, patient_ids[train_idx])
                y_pred = model.predict(X_val_scaled, patient_ids[val_idx])
            else:
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_val_scaled)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            try:
                pearson, _ = pearsonr(y_val, y_pred)
            except:
                pearson = np.nan

            records.append({
                "target": target_name,
                "model": name,
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson": pearson,
                "n_val": len(y_val)
            })

    cv_df = pd.DataFrame(records)
    # Log summary
    summary = cv_df.groupby("model").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        pearson_mean=("pearson", "mean"),
        pearson_std=("pearson", "std"),
    ).reset_index()
    summary.insert(0, "target", target_name)
    summary.insert(1, "cv_folds", n_splits)

    logger.info(f"   CV results for {target_name}: R² = {summary['r2_mean'].values[0]:.4f} ± {summary['r2_std'].values[0]:.4f}")
    return summary, cv_df