import pandas as pd
import numpy as np
import time
from pathlib import Path
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from cv import run_patient_cv
from utils import (
    calculate_comprehensive_metrics,
    create_regression_diagnostics,
    SimpleMixedEffects,
)

logger = logging.getLogger(__name__)

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict,
    targets: list[str],
    models: dict,
    output_dir: Path,
    cv_folds: int = 3,
    random_state: int = 42,
):
    """
    Train models with patient‑wise CV on train, evaluate on test.

    Args:
        train_df: Training DataFrame (includes 'id' and features).
        test_df: Test DataFrame.
        feature_sets: Dict mapping target -> list of feature column names.
        targets: List of target column names.
        models: Dict of model instances.
        output_dir: Directory to save outputs.
        cv_folds: Number of CV folds (if >1, run CV).
        random_state: Random seed.

    Returns:
        Dictionary with results per target per model.
    """
    all_results = {}
    cv_summaries = []

    for target in targets:
        if target not in train_df.columns or target not in feature_sets:
            logger.warning(f"Target {target} not available, skipping.")
            continue

        features = feature_sets[target]
        if len(features) == 0:
            logger.warning(f"No features for {target}, skipping.")
            continue

        # Drop rows missing target (necessary for supervised learning)
        train_target = train_df.dropna(subset=[target]).copy()
        test_target  = test_df.dropna(subset=[target]).copy()
        if len(train_target) == 0 or len(test_target) == 0:
            logger.warning(f"No labeled rows for {target}, skipping.")
            continue

        X_train = train_target[features]
        y_train = train_target[target].values
        X_test  = test_target[features]
        y_test  = test_target[target].values

        train_patient_ids = train_target["id"].values
        test_patient_ids  = test_target["id"].values

        logger.info(f"\n🎯 Target: {target}")
        logger.info(f"   Train: {X_train.shape} ({len(train_target['id'].unique())} patients)")
        logger.info(f"   Test:  {X_test.shape} ({len(test_target['id'].unique())} patients)")

        # ---- Patient‑wise CV on training set ----
        if cv_folds > 1:
            logger.info(f"   Running {cv_folds}-fold patient‑wise CV...")
            cv_summary, cv_fold_df = run_patient_cv(
                X_train, y_train, train_patient_ids,
                models, n_splits=cv_folds,
                target_name=target, random_state=random_state
            )
            cv_summaries.append(cv_summary)
            # Save fold‑level details
            cv_fold_df.to_csv(output_dir / f"cv_folds_{target}.csv", index=False)

        # ---- Final training on full training set and test evaluation ----
        # Preprocess on full train set
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp  = imputer.transform(X_test)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled  = scaler.transform(X_test_imp)

        target_results = {}
        for name, model in models.items():
            logger.info(f"   Training {name}...")
            start = time.time()
            if name == "MixedEffects":
                model.fit(X_train_scaled, y_train, train_patient_ids)
                y_pred_train = model.predict(X_train_scaled, train_patient_ids)
                y_pred_test  = model.predict(X_test_scaled, test_patient_ids)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test  = model.predict(X_test_scaled)
            train_time = time.time() - start

            # Compute metrics
            train_metrics = calculate_comprehensive_metrics(
                y_train, y_pred_train, f"{name} (Train)", target,
                len(features), len(train_target["id"].unique())
            )
            test_metrics = calculate_comprehensive_metrics(
                y_test, y_pred_test, f"{name} (Test)", target,
                len(features), len(test_target["id"].unique())
            )

            # Diagnostic plots
            create_regression_diagnostics(
                y_test, y_pred_test, name, target,
                len(features), len(test_target["id"].unique()),
                output_dir
            )

            # After computing y_pred_test and metrics
            target_results[name] = {
                'train_time': train_time,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model': model,
                'test_predictions': y_pred_test,  # new
                'test_ids': test_patient_ids,  # new
                'test_true': y_test,  # new
                'feature_names': features,  # new
            }
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
                target_results[name]['feature_importances'] = model.feature_importances_
            else:
                target_results[name]['feature_importances'] = None

        all_results[target] = target_results

    # Save CV summary
    if cv_summaries:
        pd.concat(cv_summaries, ignore_index=True).to_csv(
            output_dir / "cv_summary.csv", index=False
        )
        logger.info(f"CV summary saved to {output_dir / 'cv_summary.csv'}")

    return all_results