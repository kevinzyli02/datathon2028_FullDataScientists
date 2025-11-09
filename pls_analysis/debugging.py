from pathlib import Path
import pandas as pd
import numpy as np

from pls_analysis.quicktest import improved_map_symptom_values, complete_working_pipeline_fixed


def data_quality_check():
    """Check data quality before running full analysis"""
    data_directory = '/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data'
    data_path = Path(data_directory)

    print("=== DATA QUALITY CHECK ===")

    # Load data
    hormones = pd.read_csv(data_path / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(data_path / 'glucose.csv')
    sleep = pd.read_csv(data_path / 'sleep.csv')

    # Process glucose
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std']

    # Merge
    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    merged_data = pd.merge(merged_data, sleep,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')

    # Apply mapping
    merged_data = improved_map_symptom_values(merged_data)

    # Check key columns
    key_columns = ['lh', 'estrogen', 'pdg', 'appetite', 'exerciselevel', 'glucose_mean', 'minutesasleep']

    print("\nDATA QUALITY REPORT:")
    for col in key_columns:
        if col in merged_data.columns:
            non_missing = merged_data[col].notna().sum()
            total = len(merged_data)
            pct_missing = (1 - non_missing / total) * 100
            print(f"{col}: {non_missing}/{total} non-missing ({pct_missing:.1f}% missing)")

            if non_missing > 0:
                print(f"  Range: {merged_data[col].min():.2f} to {merged_data[col].max():.2f}")
                print(f"  Mean: {merged_data[col].mean():.2f}")

    # Check overlap between predictors and targets
    targets = ['lh', 'estrogen', 'pdg']
    predictors = ['appetite', 'exerciselevel', 'glucose_mean', 'minutesasleep']

    complete_cases = merged_data[targets + predictors].dropna()
    print(f"\nComplete cases for analysis: {len(complete_cases)}")

    return merged_data


# Run diagnostic first
print("Running data quality check...")
diagnostic_data = data_quality_check()

# Then run the full analysis
print("\n" + "=" * 60)
print("PROCEEDING TO FULL ANALYSIS")
print("=" * 60)
results = complete_working_pipeline_fixed()