from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


def map_symptom_values(df):
    """Convert string symptom values to numeric Likert scale (0-5)"""
    print("=== Converting symptom strings to numeric values ===")

    # Mapping dictionary based on README description
    symptom_mapping = {
        # 0-5 scale: 0 = "Not at all", 5 = "Very high"
        'Not at all': 0,
        'Very Low/Little': 1,
        'Low': 2,
        'Moderate': 3,
        'High': 4,
        'Very High': 5
    }

    # Additional mappings for flow_volume (different scale)
    flow_volume_mapping = {
        'Not at all': 0,
        'Light': 1,
        'Moderate': 2,
        'Somewhat Heavy': 3,
        'Heavy': 4,
        'Very Heavy': 5
    }

    # List of symptom columns to convert
    symptom_columns = [
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating'
    ]

    # Convert each symptom column
    for col in symptom_columns:
        if col in df.columns:
            print(f"Converting {col}...")
            # Check unique values before conversion
            unique_vals = df[col].dropna().unique()
            print(f"  Unique values in {col}: {list(unique_vals)}")

            # Map using symptom mapping
            df[col] = df[col].map(symptom_mapping)

            # Check if any values couldn't be mapped
            unmapped = df[col].isna() & df[col].notna().shift(fill_value=False)
            if unmapped.any():
                print(f"  Warning: Some values in {col} couldn't be mapped")

    # Convert flow_volume separately
    if 'flow_volume' in df.columns:
        print("Converting flow_volume...")
        unique_vals = df['flow_volume'].dropna().unique()
        print(f"  Unique values in flow_volume: {list(unique_vals)}")
        df['flow_volume'] = df['flow_volume'].map(flow_volume_mapping)

    # Convert phase to categorical codes
    if 'phase' in df.columns:
        print("Converting phase to categorical codes...")
        df['phase'] = df['phase'].astype('category').cat.codes
        df['phase'] = df['phase'].replace(-1, np.nan)  # Handle NaN in categorical

    # Convert boolean columns
    if 'is_weekend_y' in df.columns:
        print("Converting is_weekend_y to numeric...")
        df['is_weekend_y'] = df['is_weekend_y'].map({True: 1, False: 0, 'True': 1, 'False': 0})

    if 'mainsleep' in df.columns:
        print("Converting mainsleep to numeric...")
        df['mainsleep'] = df['mainsleep'].map({True: 1, False: 0, 'True': 1, 'False': 0})

    return df


def complete_working_pipeline_with_fixes():
    """Complete pipeline with proper string-to-numeric conversion"""
    data_directory = '/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data'
    data_path = Path(data_directory)

    print("=== COMPLETE WORKING PIPELINE WITH FIXES ===")

    # 1. Load data
    print("Step 1: Loading data...")
    hormones = pd.read_csv(data_path / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(data_path / 'glucose.csv')
    sleep = pd.read_csv(data_path / 'sleep.csv')

    print(f"Hormones: {hormones.shape}, Glucose: {glucose.shape}, Sleep: {sleep.shape}")

    # 2. Process glucose data
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std']

    # 3. Merge data
    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    merged_data = pd.merge(merged_data, sleep,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')

    print(f"Merged data shape: {merged_data.shape}")

    # 4. Convert string symptoms to numeric
    print("\nStep 2: Converting string values to numeric...")
    merged_data = map_symptom_values(merged_data)

    # 5. Ensure all predictors are numeric
    print("\nStep 3: Ensuring all data is numeric...")

    # Define our predictors and targets
    predictors = [
        # Symptoms (now should be numeric 0-5)
        'appetite', 'exerciselevel', 'headaches', 'cramps',
        'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
        'stress', 'foodcravings', 'indigestion', 'bloating',
        # Glucose metrics
        'glucose_mean', 'glucose_std',
        # Sleep metrics
        'minutesasleep', 'efficiency'
    ]

    targets = ['lh', 'estrogen', 'pdg']

    # Filter to available columns
    available_predictors = [col for col in predictors if col in merged_data.columns]
    available_targets = [col for col in targets if col in merged_data.columns]

    print(f"Available predictors: {available_predictors}")
    print(f"Available targets: {available_targets}")

    # Final check: convert any remaining non-numeric columns
    for col in available_predictors + available_targets:
        if merged_data[col].dtype == 'object':
            print(f"Converting {col} from object to numeric...")
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

    # 6. Run PLS analysis
    print("\nStep 4: Running PLS analysis...")
    results = safe_pls_analysis(merged_data, available_predictors, available_targets)

    if results is not None:
        print("✅ Analysis completed successfully!")
        return results
    else:
        print("❌ Analysis failed")
        return None


def safe_pls_analysis(data, predictors, targets):
    """Run PLS analysis with the specified predictors and targets"""

    # Select only the columns we need
    analysis_data = data[predictors + targets].copy()

    # Drop rows where target variables are missing
    analysis_data = analysis_data.dropna(subset=targets)
    print(f"Data after dropping missing targets: {analysis_data.shape}")

    if len(analysis_data) == 0:
        print("No data available after preprocessing")
        return None

    # Separate X and Y
    X = analysis_data[predictors]
    Y = analysis_data[targets]

    # Final check: ensure all data is numeric
    print("Final data types:")
    print(f"X dtypes: {X.dtypes}")
    print(f"Y dtypes: {Y.dtypes}")

    # Check for any non-numeric values
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            print(f"❌ ERROR: {col} is not numeric: {X[col].dtype}")
            return None

    # Handle missing values in predictors
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_imputed)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Run PLS
    print("\nRunning PLS regression...")
    n_components = min(3, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    # Calculate performance
    Y_pred_scaled = pls.predict(X_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    print("\nPLS Results:")
    r2_scores = {}
    for i, hormone in enumerate(targets):
        r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
        r2_scores[hormone] = r2
        print(f"  {hormone}: R² = {r2:.3f}")

    # Calculate feature importance (VIP scores)
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_

    p = W.shape[0]
    SS_weights = np.sum(W ** 2, axis=0)
    R2Y = np.sum(Q ** 2, axis=0)
    total_R2Y = np.sum(R2Y)

    VIP_scores = np.sqrt(p * np.sum(SS_weights * R2Y / total_R2Y * (W ** 2 / SS_weights), axis=1))

    importance_df = pd.DataFrame({
        'feature': predictors,
        'VIP_score': VIP_scores
    }).sort_values('VIP_score', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    return {
        'pls_model': pls,
        'importance_df': importance_df,
        'r2_scores': r2_scores,
        'X': X,
        'Y': Y,
        'predictors': predictors,
        'targets': targets
    }


# Run the fixed pipeline
results = complete_working_pipeline_with_fixes()

