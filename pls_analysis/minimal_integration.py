import pandas as pd
from pathlib import Path


def minimal_integration(data_directory):
    """Start with just 2-3 key datasets to avoid memory issues"""
    data_path = Path(data_directory)

    print("Loading minimal datasets for initial analysis...")

    # Load only the most essential datasets
    hormones = pd.read_csv(data_path / 'hormones_and_selfreport.csv')
    print(f"Hormones data: {hormones.shape}")

    # Add glucose data
    glucose = pd.read_csv(data_path / 'glucose.csv')
    # Aggregate glucose data by day to reduce size
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std', 'min', 'max']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max']

    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    print(f"After glucose merge: {merged_data.shape}")

    # Add sleep summary (just key columns to save memory)
    sleep = pd.read_csv(data_path / 'sleep.csv')
    sleep_reduced = sleep[
        ['id', 'sleep_start_day_in_study', 'minutesasleep', 'efficiency', 'minutestofallasleep']].copy()

    merged_data = pd.merge(merged_data, sleep_reduced,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')
    print(f"After sleep merge: {merged_data.shape}")

    return merged_data