import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import os
import gc
from sklearn.model_selection import train_test_split
import config

def load_data_polars(files, data_dir, sample_size=config.SAMPLE_SIZE, use_parquet=True):
    """
    Load all files using Polars. If use_parquet=True, looks for Parquet files in config.PARQUET_DIR.
    If Parquet file missing, falls back to CSV from data_dir.
    Aggregates large files (glucose, wrist_temperature) before merging to avoid nested object errors.
    """
    print("📂 Loading data with Polars...")

    # Determine base directory for Parquet
    parquet_dir = config.PARQUET_DIR
    parquet_dir.mkdir(exist_ok=True)

    # Start with main file
    main_file = files[0]
    if use_parquet:
        main_parquet = parquet_dir / main_file.replace('.csv', '.parquet')
        if main_parquet.exists():
            print(f"   Loading main file from Parquet: {main_parquet.resolve()}")
            df = pl.read_parquet(main_parquet)
        else:
            print(f"   Parquet not found, falling back to CSV: {data_dir / main_file}")
            df = pl.read_csv(data_dir / main_file, ignore_errors=True)
    else:
        df = pl.read_csv(data_dir / main_file, ignore_errors=True)

    df = preprocess_polars(df)
    print(f"Base data: {df.shape}")

    for file in files[1:]:
        print(f"🔗 Merging {file}...")
        try:
            # Determine file path (Parquet or CSV)
            if use_parquet:
                file_parquet = parquet_dir / file.replace('.csv', '.parquet')
                if file_parquet.exists():
                    new_df = pl.read_parquet(file_parquet)
                else:
                    print(f"   Parquet not found for {file}, falling back to CSV")
                    new_df = pl.read_csv(data_dir / file, ignore_errors=True)
            else:
                new_df = pl.read_csv(data_dir / file, ignore_errors=True)

            # Preprocess and handle file structure
            new_df = preprocess_polars(new_df)
            new_df = handle_file_structure_polars(new_df, file)

            # --- Aggregate large files BEFORE finding merge keys ---
            if file in ['glucose.csv', 'wrist_temperature.csv', 'daily_norm_unmodified.csv']:
                print(f"   📊 Aggregating {file} by patient...")
                new_df = aggregate_by_participant_polars(new_df)

            # Find appropriate merge keys (after aggregation, may only have 'id')
            merge_keys = find_merge_keys_polars(df, new_df, file)
            if not merge_keys:
                print(f"   ⚠️ No merge keys found, skipping.")
                del new_df
                continue

            # Merge
            df = df.join(new_df, on=merge_keys, how='left', suffix=f'_{file.replace(".csv","")}')
            print(f"✅ After {file}: {df.shape}")

            del new_df
            gc.collect()
        except Exception as e:
            print(f"❌ Error with {file}: {e}")
            gc.collect()

    # Sample if too large
    if df.height > sample_size:
        df = df.sample(n=sample_size, seed=config.RANDOM_STATE)
        print(f"🔽 Sampled to {df.shape}")

    return df
def preprocess_polars(df):
    """Apply symptom mapping, convert to numeric where possible, and downcast."""
    # Convert ID to integer
    if 'id' in df.columns:
        df = df.with_columns(pl.col('id').cast(pl.Int32))

    # Symptom mapping (unchanged)
    symptom_mapping = {
        'Not at all': 0, 'Very Low/Little': 1, 'Very Low': 1, 'Low': 2,
        'Moderate': 3, 'High': 4, 'Very High': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }
    symptom_cols = ['appetite','exerciselevel','headaches','cramps','sorebreasts',
                    'fatigue','sleepissue','moodswing','stress','foodcravings',
                    'indigestion','bloating']
    for col in symptom_cols:
        if col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).replace(symptom_mapping).cast(pl.Float32).alias(col))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float32))

    # --- NEW: Convert any remaining non‑numeric columns to float ---
    for col in df.columns:
        if col == 'id':
            continue
        # If column is string or object, try to cast to Float64 (coerce errors → null)
        if df[col].dtype in [pl.Utf8, pl.Object]:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
        # Downcast float64 to float32 for memory efficiency
        elif df[col].dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        # Downcast large integers
        elif df[col].dtype in [pl.Int64, pl.UInt32, pl.UInt64]:
            df = df.with_columns(pl.col(col).cast(pl.Int32))

    return df

def handle_file_structure_polars(df, file_name):
    """Rename day columns as needed."""
    if file_name in ['sleep.csv', 'computed_temperature.csv']:
        if 'sleep_start_day_in_study' in df.columns:
            df = df.rename({'sleep_start_day_in_study': 'day_in_study'})
    elif file_name == 'exercise.csv':
        if 'start_day_in_study' in df.columns:
            df = df.rename({'start_day_in_study': 'day_in_study'})
    return df

def aggregate_by_participant_polars(df):
    """Aggregate high-frequency data by patient (mean, std, min, max) for numeric columns only."""
    if 'id' not in df.columns:
        return df

    # Keep only id and columns that are numeric after conversion
    numeric_cols = [c for c in df.columns if c != 'id' and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

    if not numeric_cols:
        print(f"   ⚠️ No numeric columns to aggregate. Returning just patient IDs.")
        return df.select('id').unique()

    # Subset to only id + numeric columns to avoid any object columns
    df_numeric = df.select(['id'] + numeric_cols)

    # Create aggregation expressions
    agg_exprs = []
    for c in numeric_cols:
        agg_exprs.extend([
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).max().alias(f"{c}_max")
        ])

    agg_df = df_numeric.group_by('id').agg(agg_exprs)
    print(f"   ✅ Aggregated to {agg_df.shape}")
    return agg_df

def find_merge_keys_polars(df, new_df, file_name):
    """Find common columns suitable for merging."""
    common = set(df.columns) & set(new_df.columns)
    if 'id' in common and 'day_in_study' in common:
        return ['id', 'day_in_study']
    elif 'id' in common:
        return ['id']
    else:
        # Try alternative day columns
        day_alternatives = ['sleep_start_day_in_study', 'start_day_in_study']
        for alt in day_alternatives:
            if alt in new_df.columns and 'day_in_study' in df.columns:
                new_df = new_df.rename({alt: 'day_in_study'})
                return ['id', 'day_in_study']
    return None

# -----------------------------------------------------------------------------
# Patient split (still using pandas for simplicity)
# -----------------------------------------------------------------------------
def save_patient_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, output_dir=config.OUTPUT_DIR):
    """Save patient split using pandas (accepts Polars or pandas)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    all_patients = df['id'].unique()
    train_patients, test_patients = train_test_split(all_patients, test_size=test_size, random_state=random_state)
    split_df = pd.DataFrame({
        'patient_id': np.concatenate([train_patients, test_patients]),
        'set': ['train'] * len(train_patients) + ['test'] * len(test_patients)
    })
    split_df.to_csv(output_dir / 'patient_split.csv', index=False)
    print(f"✅ Patient split saved to {output_dir / 'patient_split.csv'}")
    return train_patients, test_patients

def load_patient_split(output_dir=config.OUTPUT_DIR):
    split_df = pd.read_csv(output_dir / 'patient_split.csv')
    train_patients = split_df[split_df['set'] == 'train']['patient_id'].values
    test_patients = split_df[split_df['set'] == 'test']['patient_id'].values
    return train_patients, test_patients