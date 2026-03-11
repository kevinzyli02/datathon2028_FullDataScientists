from pathlib import Path

BASE_DIR = Path(__file__).parent  # directory of this config file
DATA_DIR = BASE_DIR / "data"   # adjust if your data is elsewhere
OUTPUT_DIR = BASE_DIR / "patient_wise_model_analysis"
FILTERED_DATA_DIR = BASE_DIR / "filtered_data"
PARQUET_DIR = BASE_DIR / "parquet_data"
MAIN_FILE = 'hormones_and_selfreport.csv'

# Files to load
COMPREHENSIVE_FILES = [
    'hormones_and_selfreport.csv',
    'sleep.csv',
    'stress_score.csv',
    'resting_heart_rate.csv',
    'glucose.csv',                # or daily_norm_unmodified.csv – adjust as needed
    'computed_temperature.csv',
    'height_and_weight.csv',
    'exercise.csv',
    'respiratory_rate_summary.csv',
    'sleep_score.csv',
    'wrist_temperature.csv'
]

# Targets
TARGETS = ['lh', 'estrogen', 'pdg']
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_SIZE = 15000

# For stage 2 (cycle prediction)
CYCLE_TARGET = 'day_in_study'   # change if needed