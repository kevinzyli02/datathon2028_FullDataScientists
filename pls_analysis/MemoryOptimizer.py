import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc


class MemoryOptimizedMcPhasesIntegrator:
    def __init__(self, data_path):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.data = None

    def load_and_integrate_data(self, sample_fraction=1.0):
        """Load and integrate datasets with memory optimization"""
        print("Loading hormonal data...")

        # Load hormones first
        hormones_path = self.data_path / 'hormones_and_selfreport.csv'
        hormones = pd.read_csv(hormones_path)

        # Sample data if needed for memory constraints
        if sample_fraction < 1.0:
            hormones = hormones.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled {len(hormones)} rows for memory optimization")

        # Load datasets incrementally with memory cleanup
        datasets_to_load = [
            ('active_minutes', 'active_minutes.csv'),
            ('hrv', 'heart_rate_variability_details.csv'),
            ('resting_hr', 'resting_heart_rate.csv'),
            ('sleep', 'sleep.csv'),
            ('glucose', 'glucose.csv')
        ]

        for key, filename in datasets_to_load:
            try:
                print(f"Loading {filename}...")
                hormones = self._load_and_merge_safely(hormones, key, filename)
                gc.collect()  # Force garbage collection
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        self.data = hormones
        return hormones

    def _load_and_merge_safely(self, base_df, dataset_name, filename):
        """Safely load and merge one dataset at a time"""
        file_path = self.data_path / filename
        new_df = pd.read_csv(file_path)

        # Reduce memory usage by downcasting numeric types
        new_df = self._reduce_memory_usage(new_df)

        # Merge with base data
        merged = self._merge_datasets(base_df, new_df, dataset_name)

        # Clean up
        del new_df
        return merged

    def _reduce_memory_usage(self, df):
        """Reduce memory usage of DataFrame"""
        start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert object to category if few unique values
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype in ['int64', 'float64']:
                # Downcast numeric types
                if 'int' in str(df[col].dtype):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')

        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")

        return df

    def _merge_datasets(self, base_df, new_df, dataset_name):
        """Merge datasets with proper key handling"""
        # Handle different key structures
        if 'sleep_start_day_in_study' in new_df.columns:
            # Sleep-related data
            return pd.merge(base_df, new_df,
                            left_on=['id', 'day_in_study'],
                            right_on=['id', 'sleep_start_day_in_study'],
                            how='left',
                            suffixes=('', f'_{dataset_name}'))
        elif 'start_day_in_study' in new_df.columns:
            # Exercise or other start-day data
            return pd.merge(base_df, new_df,
                            left_on=['id', 'day_in_study'],
                            right_on=['id', 'start_day_in_study'],
                            how='left',
                            suffixes=('', f'_{dataset_name}'))
        else:
            # Standard daily data
            return pd.merge(base_df, new_df,
                            on=['id', 'day_in_study'],
                            how='left',
                            suffixes=('', f'_{dataset_name}'))

    def preprocess_data(self):
        """Clean and preprocess with memory optimization"""
        if self.data is None:
            raise ValueError("No data loaded.")

        df = self.data.copy()

        # Handle missing values in chunks
        print("Handling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Process in chunks if large dataset
        if len(df) > 10000:
            chunk_size = 5000
            chunks = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()
                imputer = SimpleImputer(strategy='median')
                chunk[numeric_cols] = imputer.fit_transform(chunk[numeric_cols])
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df