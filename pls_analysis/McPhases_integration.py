import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer  # This was missing!
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class McPhasesDataIntegrator:
    def __init__(self, data_path):
        # Convert to Path object if it's a string
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.data = None

    def load_and_integrate_data(self):
        """Load and integrate all relevant mcPHASES datasets"""
        print("Loading hormonal data...")

        # Base dataset with hormones and self-reports
        hormones_path = self.data_path / 'hormones_and_selfreport.csv'
        hormones = pd.read_csv(hormones_path)

        # List of datasets to integrate
        datasets_to_load = {
            'active_minutes': 'active_minutes.csv',
            'hrv': 'heart_rate_variability_details.csv',
            'resting_hr': 'resting_heart_rate.csv',
            'sleep': 'sleep.csv',
            'glucose': 'glucose.csv',
            'temperature': 'computed_temperature.csv',
            'stress': 'stress_score.csv',
            'steps': 'steps.csv',
            'demographics': 'subject-info.csv',
            'height_weight': 'height_and_weight.csv'
        }

        # Integrate each dataset
        for key, filename in datasets_to_load.items():
            file_path = self.data_path / filename
            try:
                print(f"Loading {filename}...")
                df = pd.read_csv(file_path)
                hormones = self._merge_dataset(hormones, df, key)
            except FileNotFoundError:
                print(f"Warning: {filename} not found at {file_path}, skipping...")
                continue

        self.data = hormones
        return hormones

    def _merge_dataset(self, base_df, new_df, dataset_name):
        """Merge individual datasets with the base hormonal data"""
        # Handle different key structures
        if 'sleep_start_day_in_study' in new_df.columns:
            # Sleep-related data
            merged = pd.merge(base_df, new_df,
                              left_on=['id', 'day_in_study'],
                              right_on=['id', 'sleep_start_day_in_study'],
                              how='left',
                              suffixes=('', f'_{dataset_name}'))
        elif 'start_day_in_study' in new_df.columns:
            # Exercise or other start-day data
            merged = pd.merge(base_df, new_df,
                              left_on=['id', 'day_in_study'],
                              right_on=['id', 'start_day_in_study'],
                              how='left',
                              suffixes=('', f'_{dataset_name}'))
        else:
            # Standard daily data
            merged = pd.merge(base_df, new_df,
                              on=['id', 'day_in_study'],
                              how='left',
                              suffixes=('', f'_{dataset_name}'))

        return merged

    def preprocess_data(self):
        """Clean and preprocess the integrated data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_and_integrate_data() first.")

        df = self.data.copy()

        # Handle missing values
        print("Handling missing values...")

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Impute numeric columns with median
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Create derived features
        df = self._create_derived_features(df)

        return df

    def _create_derived_features(self, df):
        """Create derived features that might be biologically relevant"""

        # Activity ratios
        if all(col in df.columns for col in ['moderately', 'lightly', 'sedentary']):
            df['moderate_to_light_ratio'] = df['moderately'] / (df['lightly'] + 1)
            df['active_to_sedentary_ratio'] = (df['moderately'] + df['lightly']) / (df['sedentary'] + 1)

        # Sleep efficiency derived
        if all(col in df.columns for col in ['minutesasleep', 'timeinbed']):
            df['sleep_efficiency_calc'] = df['minutesasleep'] / df['timeinbed']

        # BMI calculation if height and weight available
        if all(col in df.columns for col in ['height_2022', 'weight_2022']):
            df['bmi_2022'] = df['weight_2022'] / ((df['height_2022'] / 100) ** 2)

        if all(col in df.columns for col in ['height_2024', 'weight_2024']):
            df['bmi_2024'] = df['weight_2024'] / ((df['height_2024'] / 100) ** 2)

        return df