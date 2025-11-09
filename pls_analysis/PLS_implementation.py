from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

def integrate_mcphases_data():
    """
    Integrate all relevant tables for hormonal prediction
    """
    # Base hormonal data
    hormones = pd.read_csv('hormones_and_selfreport.csv')

    # Merge physiological data
    datasets = {
        'hrv': 'heart_rate_variability_details.csv',
        'sleep': 'sleep.csv',
        'activity': 'active_minutes.csv',
        'glucose': 'glucose.csv',
        'temperature': 'computed_temperature.csv',
        'stress': 'stress_score.csv'
    }

    for key, file in datasets.items():
        data = pd.read_csv(file)
        hormones = pd.merge(hormones, data,
                            on=['id', 'day_in_study'],
                            how='left',
                            suffixes=('', f'_{key}'))

    return hormones


def pls_hormone_analysis(data):
    """
    Perform PLS regression for hormonal prediction
    """
    # Separate predictors and targets
    X_columns = [
        # Physiological metrics
        'rmssd', 'low_frequency', 'high_frequency', 'resting_heart_rate',
        'minutes_asleep', 'sleep_efficiency', 'nightly_temperature',
        'glucose_value', 'steps', 'sedentary_minutes', 'moderate_minutes',
        # Symptoms
        'appetite', 'fatigue', 'stress', 'headaches', 'cramps', 'bloating'
    ]

    Y_columns = ['lh', 'estrogen', 'pdg']

    # Prepare data
    X = data[X_columns].dropna()
    Y = data[Y_columns].loc[X.index].dropna()
    X = X.loc[Y.index]  # Align indices

    # Standardize data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # PLS with cross-validation to determine components
    pls = PLSRegression(n_components=5)
    scores = cross_val_score(pls, X_scaled, Y_scaled, cv=5, scoring='neg_mean_squared_error')

    # Fit final model
    pls.fit(X_scaled, Y_scaled)

    return pls, scaler_X, scaler_Y, X_columns, Y_columns


# Analyze variable importance
def analyze_pls_components(pls_model, feature_names):
    """
    Analyze PLS component loadings and variable importance
    """
    loadings = pd.DataFrame(pls_model.x_loadings_, index=feature_names)
    VIP_scores = (pls_model.x_weights_ ** 2).sum(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'VIP_score': VIP_scores
    }).sort_values('VIP_score', ascending=False)

    return importance_df
