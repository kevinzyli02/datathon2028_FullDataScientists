# preprocessing.py
"""
Preprocess mcPHASES data: split into train/test, merge all CSVs, and validate with plots.
"""

import polars as pl
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------
# Configuration
DATA_DIR = Path(r"C:\Users\kevin\Downloads\OneDrive_2026-03-23\Edited mcphases files")
SPLIT_FILE = DATA_DIR / "McPhases SelfReport.xlsx"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------
# 1. Read train/test split from Excel
split_df = pd.read_excel(SPLIT_FILE)
# Columns: 'Patient' and 'Test/Train' (values: 'Train', 'Test')
train_ids = split_df[split_df['Test/Train'] == 'Train']['Patient'].tolist()
test_ids = split_df[split_df['Test/Train'] == 'Test']['Patient'].tolist()
print(f"Training IDs: {len(train_ids)}")
print(f"Testing IDs: {len(test_ids)}")

# ------------------------------
# 2. Load all CSV files and merge on (id, day_in_study)
# List all CSV files in DATA_DIR
csv_files = list(DATA_DIR.glob("*.csv"))
print(f"Found CSV files: {[f.name for f in csv_files]}")

# We'll read each file and keep only columns needed; all have 'id' and 'day_in_study'
# Use Polars for fast I/O and merging
def load_and_merge(csv_files, ids):
    """
    Load all CSV files for given IDs and merge them by (id, day_in_study).
    Returns a Polars DataFrame. Skips files without day_in_study column.
    """
    dfs = []
    for f in csv_files:
        # Read only necessary columns (id, day_in_study, and all other columns)
        # We'll read all columns; later we can drop duplicates
        df = pl.read_csv(f, infer_schema_length=None)
        
        # Skip files that don't have both id and day_in_study columns
        if "id" not in df.columns or "day_in_study" not in df.columns:
            print(f"Skipping {f.name}: missing id or day_in_study column")
            continue
        
        # Filter by IDs
        df = df.filter(pl.col("id").is_in(ids))
        dfs.append(df)
    # Merge all DataFrames on id and day_in_study
    # Use a left join chain; first DataFrame is the base
    if not dfs:
        return pl.DataFrame()
    base = dfs[0]
    for idx, df in enumerate(dfs[1:], start=1):
        base = base.join(df, on=["id", "day_in_study"], how="left", suffix=f"_{idx}")
    return base

# Load train and test data
train_df = load_and_merge(csv_files, train_ids)
test_df = load_and_merge(csv_files, test_ids)

# Save as Parquet for efficient storage
train_df.write_parquet(OUTPUT_DIR / "train_data.parquet")
test_df.write_parquet(OUTPUT_DIR / "test_data.parquet")
print(f"Saved train data shape: {train_df.shape}")
print(f"Saved test data shape: {test_df.shape}")

# ------------------------------
# 3. Validate data by recreating figures from mcphases_sample_analysis.py
# Convert to pandas for plotting
train_pd = train_df.to_pandas()
test_pd = test_df.to_pandas()
all_data = pd.concat([train_pd, test_pd], ignore_index=True)

# --- 3.1 Hormone curves across cycle (only for training set, as in original script)
# Use only the main dataset (hormones_and_selfreport.csv) for this
# We need to ensure we have 'phase', 'lh', 'estrogen', 'pdg' columns.
# The original script used 'hr_and_selfreport' which is hormones_and_selfreport.csv.
# Our merged train data includes those columns.

# Rename columns if necessary (original used 'estrogen', 'lh', 'pdg')
# Make sure phase is categorical
hormone_cols = ['lh', 'estrogen', 'pdg']
if all(col in train_pd.columns for col in hormone_cols):
    # Prepare data: compute menstrual cycle starts and normalize
    # (Exactly as in original script)
    hr = train_pd.copy()
    hr['new_cycle'] = ((hr['phase'] == 'Menstrual') &
                       (hr['phase'].shift(1) != 'Menstrual')).astype(bool)
    hr['cycle_menstrual'] = hr.groupby('id')['new_cycle'].cumsum()
    hr.drop('new_cycle', axis=1, inplace=True)

    # Normalize cycle length
    hr['cycle_length'] = hr.groupby(['id', 'cycle_menstrual'])['phase'].transform('count')
    hr['row_in_cycle'] = hr.groupby(['id', 'cycle_menstrual']).cumcount() + 1
    hr['percent'] = (hr['row_in_cycle'] / hr['cycle_length']) * 100
    hr.drop(['cycle_length', 'row_in_cycle'], axis=1, inplace=True)

    # Keep only middle cycles per person
    filtered = []
    for _, g in hr.groupby('id'):
        minc = g['cycle_menstrual'].min()
        maxc = g['cycle_menstrual'].max()
        mid = g[(g['cycle_menstrual'] != minc) & (g['cycle_menstrual'] != maxc)]
        filtered.append(mid)
    one_cycle_df = pd.concat(filtered)

    # Interpolation function
    def interpolate(group, signal, grid=np.linspace(0,100,100)):
        if len(group) >= 2:
            f = interp1d(group['percent'], group[signal], kind='linear', fill_value='extrapolate')
            return f(grid)
        return np.full_like(grid, np.nan)

    common_grid = np.linspace(0,100,100)
    averaged = {}
    sem = {}
    for sig in hormone_cols:
        interp_vals = []
        for (pid, cycle), g in one_cycle_df.groupby(['id', 'cycle_menstrual']):
            vals = interpolate(g, sig)
            interp_vals.append(vals)
        interp_vals = np.vstack(interp_vals)
        averaged[sig] = np.nanmean(interp_vals, axis=0)
        sem[sig] = np.nanstd(interp_vals, axis=0) / np.sqrt(np.sum(~np.isnan(interp_vals), axis=0))

    # Plot
    fig, axes = plt.subplots(3,1, figsize=(8,8), sharex=True)
    for i, sig in enumerate(hormone_cols):
        smoothed = lowess(averaged[sig], common_grid, frac=0.3)[:,1]
        ci = lowess(sem[sig], common_grid, frac=0.3)[:,1]
        axes[i].plot(common_grid, smoothed, color=f'C{i}')
        axes[i].fill_between(common_grid, smoothed-1.96*ci, smoothed+1.96*ci, alpha=0.3, color=f'C{i}')
        axes[i].set_ylabel(sig.upper())
        axes[i].grid(False)
    axes[-1].set_xlabel('Normalized Menstrual Cycle (%)')
    for v in [20,40,60,80]:
        for ax in axes:
            ax.axvline(v, color='gray', linestyle='--', alpha=0.5)
    plt.suptitle('Hormone trajectories across cycle (training data)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hormone_curves.png')
    plt.show()
else:
    print("Hormone columns not found, skipping hormone curves plot")

# --- 3.2 Boxplots of RHR and temperature across phases (using all data)
# Need 'phase', 'value' (RHR), 'nightly_temperature'
if 'value' in all_data.columns and 'nightly_temperature' in all_data.columns:
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    # Temperature
    temp_vals = []
    temp_labels = []
    for source in ['train', 'test']:
        df = train_pd if source == 'train' else test_pd
        for phase in ['Menstrual','Follicular','Fertility','Luteal']:
            vals = df[df['phase']==phase]['nightly_temperature'].dropna()
            if len(vals) > 0:
                temp_vals.append(vals)
                temp_labels.append(f'{phase}\n{source}')
    axs[0].boxplot(temp_vals)
    axs[0].set_xticklabels(temp_labels, rotation=45, ha='right')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Nightly Temperature by Phase')

    # RHR
    rhr_vals = []
    rhr_labels = []
    for source in ['train', 'test']:
        df = train_pd if source == 'train' else test_pd
        for phase in ['Menstrual','Follicular','Fertility','Luteal']:
            vals = df[df['phase']==phase]['value'].dropna()
            if len(vals) > 0:
                rhr_vals.append(vals)
                rhr_labels.append(f'{phase}\n{source}')
    axs[1].boxplot(rhr_vals)
    axs[1].set_xticklabels(rhr_labels, rotation=45, ha='right')
    axs[1].set_ylabel('RHR (bpm)')
    axs[1].set_title('Resting Heart Rate by Phase')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'physio_by_phase.png')
    plt.show()
else:
    print("RHR or temperature columns not found, skipping boxplots")

print("Preprocessing complete.")