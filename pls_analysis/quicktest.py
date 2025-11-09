from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


def improved_map_symptom_values(df):
    """Improved mapping that handles all the unique values we observed"""
    print("=== Converting symptom strings to numeric values ===")

    # More comprehensive mapping based on the actual data we saw
    comprehensive_mapping = {
        # 0-5 scale
        'Not at all': 0,
        'Very Low/Little': 1,
        'Very Low': 1,
        'Low': 2,
        'Moderate': 3,
        'High': 4,
        'Very High': 5,
        # Handle numeric strings that might be present
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
    }

    # Additional mappings for specific columns
    flow_volume_mapping = {
        'Not at all': 0,
        'Spotting / Very Light': 1,
        'Light': 2,
        'Somewhat Light': 2,
        'Moderate': 3,
        'Somewhat Heavy': 4,
        'Heavy': 5,
        'Very Heavy': 6
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

            # Map using comprehensive mapping
            df[col] = df[col].map(comprehensive_mapping)

            # Check results
            mapped_vals = df[col].dropna().unique()
            print(f"  Mapped to: {list(mapped_vals)}")

    # Convert flow_volume separately
    if 'flow_volume' in df.columns:
        print("Converting flow_volume...")
        unique_vals = df['flow_volume'].dropna().unique()
        print(f"  Unique values in flow_volume: {list(unique_vals)}")
        df['flow_volume'] = df['flow_volume'].map(flow_volume_mapping)
        print(f"  Mapped to: {list(df['flow_volume'].dropna().unique())}")

    # Convert phase to categorical codes
    if 'phase' in df.columns:
        print("Converting phase to categorical codes...")
        df['phase'] = df['phase'].astype('category').cat.codes
        df['phase'] = df['phase'].replace(-1, np.nan)
        print(f"  Mapped to: {list(df['phase'].dropna().unique())}")

    # Convert boolean columns
    bool_columns = ['is_weekend_y', 'mainsleep']
    for col in bool_columns:
        if col in df.columns:
            print(f"Converting {col} to numeric...")
            # Handle both boolean and string representations
            df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0})
            print(f"  Mapped to: {list(df[col].dropna().unique())}")

    return df


def robust_pls_analysis(data, predictors, targets):
    """Robust PLS analysis that handles missing data and array length issues"""

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

    # Check for columns with no observed values
    print("\nChecking for columns with no observed values:")
    columns_with_data = []
    for col in X.columns:
        non_missing_count = X[col].notna().sum()
        print(f"  {col}: {non_missing_count} non-missing values")
        if non_missing_count > 0:
            columns_with_data.append(col)

    # Use only columns that have data
    X = X[columns_with_data]
    print(f"\nUsing {len(columns_with_data)} predictors with data: {columns_with_data}")

    if len(columns_with_data) == 0:
        print("No predictors with data available")
        return None

    # Final check: ensure all data is numeric
    print("\nFinal data types:")
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            print(f"  Converting {col} to numeric...")
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Handle missing values in predictors - only for columns with data
    print("\nImputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Convert back to DataFrame to maintain column names
    X_imputed_df = pd.DataFrame(X_imputed, columns=columns_with_data, index=X.index)

    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_imputed_df)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Run PLS
    print("\nRunning PLS regression...")
    n_components = min(3, X_scaled.shape[1], X_scaled.shape[0] - 1)
    print(f"Using {n_components} PLS components")

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

    # Calculate feature importance (VIP scores) - FIXED VERSION
    try:
        W = pls.x_weights_
        Q = pls.y_loadings_

        p = W.shape[0]  # number of features
        SS_weights = np.sum(W ** 2, axis=0)
        R2Y = np.sum(Q ** 2, axis=0)
        total_R2Y = np.sum(R2Y)

        # Calculate VIP scores correctly
        VIP_scores = np.sqrt(p * np.sum((W ** 2) * (R2Y / total_R2Y), axis=1))

        # Create importance dataframe with the correct features
        importance_df = pd.DataFrame({
            'feature': columns_with_data,
            'VIP_score': VIP_scores
        }).sort_values('VIP_score', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

    except Exception as e:
        print(f"Error calculating VIP scores: {e}")
        # Fallback: use absolute loadings as importance
        loadings = pd.DataFrame(pls.x_loadings_, index=columns_with_data)
        importance_scores = loadings.abs().mean(axis=1)
        importance_df = pd.DataFrame({
            'feature': importance_scores.index,
            'VIP_score': importance_scores.values
        }).sort_values('VIP_score', ascending=False)
        print("\nTop 10 Most Important Features (using loadings):")
        print(importance_df.head(10))

    return {
        'pls_model': pls,
        'importance_df': importance_df,
        'r2_scores': r2_scores,
        'X': X_imputed_df,
        'Y': Y,
        'predictors': columns_with_data,
        'targets': targets
    }


def complete_working_pipeline_fixed():
    """Complete pipeline with robust error handling"""
    data_directory = '/Users/kevin/Documents/GitHub/datathon2028_FullDataScientists/data'
    data_path = Path(data_directory)

    print("=== COMPLETE WORKING PIPELINE - FIXED VERSION ===")

    # 1. Load data
    print("Step 1: Loading data...")
    hormones = pd.read_csv(data_path / 'hormones_and_selfreport.csv')
    glucose = pd.read_csv(data_path / 'glucose.csv')
    sleep = pd.read_csv(data_path / 'sleep.csv')

    print(f"Hormones: {hormones.shape}, Glucose: {glucose.shape}, Sleep: {sleep.shape}")

    # 2. Process glucose data - handle potential memory issues
    print("Processing glucose data...")
    glucose_daily = glucose.groupby(['id', 'day_in_study']).agg({
        'glucose_value': ['mean', 'std']
    }).reset_index()
    glucose_daily.columns = ['id', 'day_in_study', 'glucose_mean', 'glucose_std']

    # 3. Merge data
    print("Merging data...")
    merged_data = pd.merge(hormones, glucose_daily, on=['id', 'day_in_study'], how='left')
    merged_data = pd.merge(merged_data, sleep,
                           left_on=['id', 'day_in_study'],
                           right_on=['id', 'sleep_start_day_in_study'],
                           how='left')

    print(f"Merged data shape: {merged_data.shape}")

    # 4. Convert string symptoms to numeric
    print("\nStep 2: Converting string values to numeric...")
    merged_data = improved_map_symptom_values(merged_data)

    # 5. Ensure all predictors are numeric
    print("\nStep 3: Ensuring all data is numeric...")

    # Define our predictors and targets
    predictors = [
        # Symptoms
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

    # 6. Run robust PLS analysis
    print("\nStep 4: Running robust PLS analysis...")
    results = robust_pls_analysis(merged_data, available_predictors, available_targets)

    if results is not None:
        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        # Print summary
        print("\nSUMMARY:")
        print(f"Number of observations: {results['X'].shape[0]}")
        print(f"Number of predictors used: {len(results['predictors'])}")
        print(f"R² Scores:")
        for hormone, r2 in results['r2_scores'].items():
            print(f"  {hormone}: {r2:.3f}")

        return results
    else:
        print("❌ Analysis failed")
        return None


# Run the fixed pipeline
results = complete_working_pipeline_fixed()