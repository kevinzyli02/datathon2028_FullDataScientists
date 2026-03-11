from pathlib import Path

# Paths
DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data")
OUTPUT_DIR = Path(r'C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\Combined Model\patient_wise_model_analysis')
FILTERED_DATA_DIR = Path('filtered_data')
PARQUET_DIR = Path('parquet_data')          # new: for converted Parquet files
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