#!/usr/bin/env python3
"""
Refactored hormone prediction pipeline – final version.

Features:
- Predefined patient split from Excel.
- Patient‑wise 3‑fold CV on training set.
- Baseline normalisation and personalised features applied separately to train/test.
- Lagged features (previous days) of daily sensor aggregates.
- Predicts both raw and normalized versions of each hormone.
- Diagnostic prints for hormone non‑null counts.
- Post‑training outputs: predictions CSV, residual plots, feature importance Excel.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
# Add parent directory to path if running as script
sys.path.append(str(Path(__file__).parent.parent))

# Force matplotlib to use non‑interactive backend (prevents Tkinter thread errors)
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
    SimpleMixedEffects,
    RandomForestRegressor,
    xgb,
    lgb,
    create_comprehensive_report,
    generate_post_training_outputs,
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

        # 3. Apply baseline normalisation and personalised features SEPARATELY
        logger.info("🔄 Applying baseline normalisation (per set)...")
        train_norm = normalize_by_baseline(train_raw, TARGETS, config.baseline_days)
        test_norm  = normalize_by_baseline(test_raw, TARGETS, config.baseline_days)

        # Diagnostic prints after normalisation
        print("DEBUG: pdg_normalized non-null count in train_norm:", train_norm['pdg_normalized'].notna().sum())
        print("DEBUG: pdg_normalized non-null count in test_norm:", test_norm['pdg_normalized'].notna().sum())

        logger.info("🎯 Adding personalised features (per set)...")
        train_enh = add_personalized_features(train_norm, TARGETS, config.rolling_window)
        test_enh  = add_personalized_features(test_norm, TARGETS, config.rolling_window)

        # Add lagged features (previous days)
        logger.info("⏪ Adding lagged features (previous days)...")
        train_enh = add_lag_features(train_enh, lag_days=[1,2,3])
        test_enh  = add_lag_features(test_enh, lag_days=[1,2,3])

        # Combine for feature detection (columns must be identical)
        combined = pd.concat([train_enh, test_enh], ignore_index=True)

        # Build list of all possible target columns (raw and normalized)
        possible_targets = []
        for raw_target in TARGETS:
            # Raw version
            if raw_target in combined.columns and combined[raw_target].notna().sum() > 0:
                possible_targets.append(raw_target)
                print(f"Will predict raw: {raw_target}")
            # Normalized version
            norm_col = f"{raw_target}_normalized"
            if norm_col in combined.columns and combined[norm_col].notna().sum() > 0:
                possible_targets.append(norm_col)
                print(f"Will predict normalized: {norm_col}")

        logger.info(f"All targets to predict: {possible_targets}")

        # Build feature sets for each target (automatically excludes other hormone columns)
        feature_sets = get_all_features(combined, possible_targets)

        # 4. Define models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=config.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=config.random_state, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=config.random_state, n_jobs=-1, verbose=-1
            ),
            'MixedEffects': SimpleMixedEffects(min_patient_samples=5),
        }

        # 5. Train models with CV on train, evaluate on test
        results = train_and_evaluate(
            train_enh, test_enh, feature_sets, possible_targets, models,
            output_dir=config.output_dir, cv_folds=config.cv_folds,
            random_state=config.random_state
        )

        # 6. Generate final comprehensive report (original function)
        create_comprehensive_report(results, feature_sets)

        # 7. Generate post‑training outputs: predictions CSV, residual plots, feature importance Excel
        logger.info("📊 Generating post‑training outputs...")
        generate_post_training_outputs(results, config.output_dir)

    else:
        # Fallback to original random‑split pipeline (not shown here for brevity)
        logger.info("Using original random split (not recommended for final analysis)")
        # ... call original main_with_filtering_memory_efficient ...

    logger.info(f"✅ Pipeline finished. Results in {config.output_dir}")

if __name__ == "__main__":
    main()