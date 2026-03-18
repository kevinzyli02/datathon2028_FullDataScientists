#!/usr/bin/env python3
"""
Refactored hormone prediction pipeline – final version with lag comparison.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from splits import load_predefined_split, apply_split, write_split_manifest
from modeling import train_and_evaluate
from utils import (
    load_comprehensive_data_memory_efficient,
    normalize_by_baseline,
    add_personalized_features,
    add_lag_features,
    get_all_features,
    TARGETS,
    COMPREHENSIVE_FILES,
    RandomForestRegressor,
    xgb,
    create_comprehensive_report,
    generate_post_training_outputs,
    run_shap_analysis,          # <-- ensure imported
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    config = Config.from_args()
    config.output_dir.mkdir(exist_ok=True)

    logger.info("🚀 Starting refactored hormone prediction pipeline")
    logger.info(f"Config: {config}")

    # 1. Load raw data (all files, left merges)
    logger.info("📂 Loading comprehensive dataset...")
    df_raw = load_comprehensive_data_memory_efficient(
        COMPREHENSIVE_FILES, config.data_dir, config.sample_size
    )
    logger.info(f"Raw dataset shape: {df_raw.shape}")

    # Diagnostic prints for raw hormone counts
    print("DEBUG: pdg non-null count in raw data:", df_raw['pdg'].notna().sum())
    print("DEBUG: lh non-null count in raw data:", df_raw['lh'].notna().sum())
    print("DEBUG: estrogen non-null count in raw data:", df_raw['estrogen'].notna().sum())

    # 2. Split data according to predefined split (if requested)
    if config.use_predefined_split:
        logger.info("📋 Using predefined patient split from Excel")
        train_ids, test_ids = load_predefined_split(
            config.selfreport_path, config.summary_sheet
        )

        # Apply split BEFORE any patient‑wise feature engineering
        train_raw, test_raw = apply_split(df_raw, train_ids, test_ids)

        # Write manifest
        write_split_manifest(train_ids, test_ids, df_raw, TARGETS, config.output_dir)
        logger.info(f"Train patients: {len(train_ids)}, Test patients: {len(test_ids)}")

        # 3. Apply baseline normalisation and personalised features (common to both experiments)
        logger.info("🔄 Applying baseline normalisation (per set)...")
        train_norm = normalize_by_baseline(train_raw, TARGETS, config.baseline_days)
        test_norm  = normalize_by_baseline(test_raw, TARGETS, config.baseline_days)

        logger.info("🎯 Adding personalised features (per set)...")
        train_enh_base = add_personalized_features(train_norm, TARGETS, config.rolling_window)
        test_enh_base  = add_personalized_features(test_norm, TARGETS, config.rolling_window)

        # 4. Define lag configurations to compare
        lag_configs = [
            {"name": "no_lag", "lag_days": []},
            {"name": "with_lag", "lag_days": [1, 2, 3]},
        ]

        for lag_cfg in lag_configs:
            logger.info(f"\n🔬 Running experiment: {lag_cfg['name']}")
            exp_output_dir = config.output_dir / lag_cfg['name']
            exp_output_dir.mkdir(exist_ok=True)

            # Apply lag features if requested
            if lag_cfg['lag_days']:
                train_exp = add_lag_features(train_enh_base.copy(), lag_days=lag_cfg['lag_days'])
                test_exp  = add_lag_features(test_enh_base.copy(), lag_days=lag_cfg['lag_days'])
            else:
                train_exp = train_enh_base.copy()
                test_exp  = test_enh_base.copy()

            # Combine for feature detection
            combined = pd.concat([train_exp, test_exp], ignore_index=True)

            # Build list of all possible target columns (raw and normalized)
            possible_targets = []
            for raw_target in TARGETS:
                if raw_target in combined.columns and combined[raw_target].notna().sum() > 0:
                    possible_targets.append(raw_target)
                    print(f"Will predict raw: {raw_target}")
                norm_col = f"{raw_target}_normalized"
                if norm_col in combined.columns and combined[norm_col].notna().sum() > 0:
                    possible_targets.append(norm_col)
                    print(f"Will predict normalized: {norm_col}")

            logger.info(f"All targets to predict: {possible_targets}")

            # Build feature sets for each target
            feature_sets = get_all_features(combined, possible_targets)

            # 5. Define models (only RandomForest and XGBoost as requested)
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=config.random_state, n_jobs=-1
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=config.random_state, n_jobs=-1
                ),
            }

            # 6. Train models with CV on train, evaluate on test
            results = train_and_evaluate(
                train_exp, test_exp, feature_sets, possible_targets, models,
                output_dir=exp_output_dir, cv_folds=config.cv_folds,
                random_state=config.random_state
            )

            # 7. Generate final comprehensive report
            create_comprehensive_report(results, feature_sets)

            # 8. Generate post‑training outputs: predictions CSV, residual plots, feature importance Excel
            logger.info("📊 Generating post‑training outputs...")
            generate_post_training_outputs(results, exp_output_dir)

            # 9. Run SHAP analysis on the best model for each target
            logger.info("🔍 Running SHAP analysis...")
            run_shap_analysis(results, feature_sets, exp_output_dir, sample_size=30)

        logger.info(f"✅ All experiments finished. Results in {config.output_dir}")

    else:
        # Fallback to original random‑split pipeline (not shown for brevity)
        logger.info("Using original random split (not recommended for final analysis)")
        # ... call original main_with_filtering_memory_efficient ...

if __name__ == "__main__":
    main()