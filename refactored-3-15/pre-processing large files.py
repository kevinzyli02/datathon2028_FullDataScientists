"""
Process raw data files into a daily‑aggregated format.
Run this script once to create a 'processed_data' folder with downsized files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import gc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RAW_DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")                 # where your original files are
PROCESSED_DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data\processed data") # where processed files will go

# Files that are already at daily level (or should not be aggregated)
# These will be copied as‑is.
DAILY_FILES = {
    "hormones_and_selfreport.csv",
    "daily_norm.csv",
    "height_and_weight.csv",   # one row per patient, not per day? but keep as is
    # add any others that are already daily
}

# Files that contain high‑frequency data and should be aggregated to daily.
# The script will also treat any file that has more than 100 rows per patient
# (on average) as high‑frequency, but you can force it with this set.
HIGH_FREQ_FILES = {
    "sleep.csv",
    "stress_score.csv",
    "resting_heart_rate.csv",
    "computed_temperature.csv",
    "exercise.csv",
    "respiratory_rate_summary.csv",
    "sleep_score.csv",
    "wrist_temperature.csv",
}

# If a file has more than this many rows per patient (on average), it will be
# aggregated even if not listed in HIGH_FREQ_FILES.
ROW_PER_PATIENT_THRESHOLD = 5

# Day column names to look for (in order of preference)
DAY_COLUMN_NAMES = ["day_in_study", "sleep_start_day_in_study", "start_day_in_study"]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def find_day_column(df):
    """Return the name of the column that represents the day in study."""
    for col in DAY_COLUMN_NAMES:
        if col in df.columns:
            return col
    return None

def aggregate_to_daily(df, file_name):
    """
    Aggregate a high‑frequency DataFrame to daily level.
    Returns a DataFrame with one row per (id, day).
    """
    if "id" not in df.columns:
        print(f"   ⚠️ No 'id' column in {file_name}, cannot aggregate. Skipping.")
        return None

    day_col = find_day_column(df)
    if day_col is None:
        print(f"   ⚠️ No day column found in {file_name}, cannot aggregate. Skipping.")
        return None

    # Identify numeric columns to aggregate (exclude id and day column)
    exclude = {"id", day_col}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not numeric_cols:
        print(f"   ⚠️ No numeric columns to aggregate in {file_name}. Copying as is.")
        return df

    # Group by id and day, compute stats
    grouped = df.groupby(["id", day_col])[numeric_cols].agg(["mean", "std", "min", "max", "count"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]  # flatten
    grouped = grouped.reset_index()

    print(f"   Aggregated from {len(df)} rows to {len(grouped)} rows.")
    return grouped

# -----------------------------------------------------------------------------
# Main processing loop
# -----------------------------------------------------------------------------
def main():
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    # List all files to process (you can customise this list)
    all_files = list(RAW_DATA_DIR.glob("*.csv"))
    # Optionally filter by the list you know you have
    # all_files = [RAW_DATA_DIR / f for f in HIGH_FREQ_FILES.union(DAILY_FILES) if (RAW_DATA_DIR / f).exists()]

    print(f"Found {len(all_files)} CSV files in {RAW_DATA_DIR}")

    for file_path in all_files:
        file_name = file_path.name
        print(f"\n📄 Processing {file_name}...")

        # Skip if file is empty or we can't read it
        if file_path.stat().st_size == 0:
            print("   ⚠️ File is empty, skipping.")
            continue

        try:
            # Read the file (low_memory=False to avoid dtype warnings)
            df = pd.read_csv(file_path, low_memory=False)
            original_rows = len(df)
            print(f"   Original rows: {original_rows}")

            # Determine if this file should be aggregated
            aggregate = False
            if file_name in HIGH_FREQ_FILES:
                aggregate = True
                print("   High‑frequency file (by name).")
            elif file_name not in DAILY_FILES:
                # Check average rows per patient as a heuristic
                if "id" in df.columns:
                    patients = df["id"].nunique()
                    if patients > 0 and original_rows / patients > ROW_PER_PATIENT_THRESHOLD:
                        aggregate = True
                        print(f"   High‑frequency heuristic triggered: {original_rows/patients:.1f} rows/patient.")
                else:
                    print("   No 'id' column, cannot check frequency. Copying as is.")

            if aggregate:
                processed_df = aggregate_to_daily(df, file_name)
                if processed_df is None:
                    # fallback: copy original
                    processed_df = df
            else:
                processed_df = df
                print("   File is daily‑level, copying as is.")

            # Save to processed folder
            out_path = PROCESSED_DATA_DIR / file_name
            processed_df.to_csv(out_path, index=False)
            print(f"   💾 Saved to {out_path} ({len(processed_df)} rows)")

            # Clean up
            del df, processed_df
            gc.collect()

        except Exception as e:
            print(f"   ❌ Error processing {file_name}: {e}")

    print(f"\n✅ Processing complete. Files saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()