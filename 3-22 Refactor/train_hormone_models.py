# hormone_model.py
"""
Train models to predict hormone levels (LH, estrogen, PdG) from Fitbit data.
Uses XGBoost, Random Forest, and LSTM/GRU.
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data\processed")
TRAIN_FILE = DATA_DIR / "train_data.parquet"
TEST_FILE = DATA_DIR / "test_data.parquet"
OUTPUT_DIR = DATA_DIR / "hormone_models"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------
# 1. Load data
train = pl.read_parquet(TRAIN_FILE).to_pandas()
test = pl.read_parquet(TEST_FILE).to_pandas()

# 2. Feature engineering
# We'll use all available columns except the hormone targets and ID/time columns.
# For simplicity, we'll exclude: id, day_in_study, phase, cycle_menstrual, etc.
# Also drop any columns that are not numeric or are target-related.
target_cols = ['lh', 'estrogen', 'pdg']  # may need to match exact names in data
# Find which of these exist in data
targets = [c for c in target_cols if c in train.columns]
print(f"Target columns found: {targets}")

# Exclude columns that are not features
exclude_cols = ['id', 'day_in_study', 'phase', 'study_interval'] + targets
feature_cols = [c for c in train.columns if c not in exclude_cols and train[c].dtype in ['float64', 'int64']]
print(f"Feature columns: {feature_cols}")

# Handle missing values: simple forward fill per patient, then median fill
train = train.sort_values(['id', 'day_in_study'])
test = test.sort_values(['id', 'day_in_study'])
for col in feature_cols:
    train[col] = train.groupby('id')[col].fillna(method='ffill')
    test[col] = test.groupby('id')[col].fillna(method='ffill')
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# Separate features and targets
X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train = {t: train[t].values for t in targets}
y_test = {t: test[t].values for t in targets}

# Scale features (for tree models, scaling not required; for NN it is)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define models and evaluation functions
def evaluate_regression(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return r2, mae, rmse

# 4. XGBoost and Random Forest for each target
results = {}
for target in targets:
    print(f"\n=== Predicting {target} ===")
    yt = y_train[target]
    yt_test = y_test[target]
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, yt)
    pred_xgb = xgb_model.predict(X_test)
    r2, mae, rmse = evaluate_regression(yt_test, pred_xgb, "XGBoost")
    results[(target, 'XGBoost')] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'model': xgb_model, 'pred': pred_xgb}
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, yt)
    pred_rf = rf_model.predict(X_test)
    r2, mae, rmse = evaluate_regression(yt_test, pred_rf, "Random Forest")
    results[(target, 'RF')] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'model': rf_model, 'pred': pred_rf}

# 5. LSTM/GRU (multi-output)
# We need sequences: each patient's time series. Since we have day_in_study, we can create sequences of length 7 (or variable).
# We'll create sequences for each patient separately.
def create_sequences(X, y_dict, seq_len=7):
    X_seq, y_seq = [], []
    for pid, group in train.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        y_p = np.array([group[t].values for t in targets]).T  # shape (n_days, n_targets)
        for i in range(len(X_p) - seq_len):
            X_seq.append(X_p[i:i+seq_len])
            y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# For testing, we need to align sequences; we can create sequences using the same day alignment.
# To simplify, we'll create sequences from training set and then use the same patient order for test?
# But test set may have different patients. Better: create sequences from both sets separately,
# but ensure we use same feature scaling. We'll create sequences from test set after scaling.

# Build sequences
seq_len = 7
X_train_seq, y_train_seq = create_sequences(train, y_train, seq_len)
# For test, we need to create sequences from test data using the same scaling.
# Since test set may have different patients, we'll create test sequences similarly.
def create_sequences_test(X_df, y_dict, seq_len, scaler, feature_cols):
    X_seq, y_seq = [], []
    for pid, group in X_df.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        X_p = scaler.transform(X_p)
        y_p = np.array([group[t].values for t in targets]).T
        for i in range(len(X_p) - seq_len):
            X_seq.append(X_p[i:i+seq_len])
            y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# For training, we already have scaled features; we need to scale the training sequences as well.
# But X_train_seq currently uses unscaled features. So we need to scale after creating sequences.
# Alternatively, we can scale before creating sequences. Let's scale the whole train set features,
# then create sequences.
train_scaled = scaler.transform(train[feature_cols].values)
train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols)
train_scaled_df['id'] = train['id'].values
train_scaled_df['day_in_study'] = train['day_in_study'].values
# Now create sequences from this scaled DataFrame
def create_sequences_scaled(df, y_dict, seq_len):
    X_seq, y_seq = [], []
    for pid, group in df.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        y_p = np.array([train[t].loc[group.index].values for t in targets]).T
        for i in range(len(X_p) - seq_len):
            X_seq.append(X_p[i:i+seq_len])
            y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences_scaled(train_scaled_df, y_train, seq_len)
# For test
test_scaled = scaler.transform(test[feature_cols].values)
test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols)
test_scaled_df['id'] = test['id'].values
test_scaled_df['day_in_study'] = test['day_in_study'].values
X_test_seq, y_test_seq = create_sequences_scaled(test_scaled_df, y_test, seq_len)

print(f"Train sequences shape: {X_train_seq.shape}, Test sequences shape: {X_test_seq.shape}")

# Build LSTM/GRU model
def build_lstm(input_shape, n_outputs):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(n_outputs)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape, n_outputs):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(n_outputs)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# We'll train LSTM and GRU models
n_outputs = len(targets)
input_shape = (seq_len, len(feature_cols))

lstm_model = build_lstm(input_shape, n_outputs)
gru_model = build_gru(input_shape, n_outputs)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train LSTM
history_lstm = lstm_model.fit(X_train_seq, y_train_seq, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_lstm = lstm_model.predict(X_test_seq)
for i, t in enumerate(targets):
    r2, mae, rmse = evaluate_regression(y_test_seq[:, i], pred_lstm[:, i], f"LSTM-{t}")
    results[(t, 'LSTM')] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'model': lstm_model, 'pred': pred_lstm[:, i]}

# Train GRU
history_gru = gru_model.fit(X_train_seq, y_train_seq, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_gru = gru_model.predict(X_test_seq)
for i, t in enumerate(targets):
    r2, mae, rmse = evaluate_regression(y_test_seq[:, i], pred_gru[:, i], f"GRU-{t}")
    results[(t, 'GRU')] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'model': gru_model, 'pred': pred_gru[:, i]}

# 6. Create scatter plots and residual plots for each model and target
for target in targets:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    models = ['XGBoost', 'RF', 'LSTM', 'GRU']
    # We'll create separate plots for each model
    for idx, model_name in enumerate(models):
        if (target, model_name) in results:
            pred = results[(target, model_name)]['pred']
            y_true = y_test[target] if model_name in ['XGBoost', 'RF'] else y_test_seq[:, targets.index(target)]
            # Scatter plot (actual vs predicted)
            ax_scatter = axes[0, idx]
            ax_scatter.scatter(y_true, pred, alpha=0.5, c='blue')
            ax_scatter.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            ax_scatter.set_xlabel('Actual')
            ax_scatter.set_ylabel('Predicted')
            ax_scatter.set_title(f'{model_name} - {target}')
            # Residual plot (predicted vs residual)
            ax_res = axes[1, idx]
            residuals = y_true - pred
            ax_res.scatter(pred, residuals, alpha=0.5, c='green')
            ax_res.axhline(0, color='red', linestyle='--')
            ax_res.set_xlabel('Predicted')
            ax_res.set_ylabel('Residual')
            ax_res.set_title(f'Residuals - {model_name}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{target}_predictions.png')
    plt.show()

# 7. Feature importance table for XGBoost and Random Forest
# We'll collect feature importances per model and categorize them.
# Need a mapping of feature categories. We'll assume we have a list of feature names and categories.
# For demonstration, we'll create dummy categories based on feature name patterns.
def categorize_feature(feature_name):
    if 'sleep' in feature_name or 'awake' in feature_name:
        return 'sleep'
    elif 'temp' in feature_name or 'temperature' in feature_name:
        return 'temperature'
    elif 'hr' in feature_name or 'heart' in feature_name:
        return 'heart_rate'
    elif 'step' in feature_name or 'activity' in feature_name:
        return 'activity'
    elif 'stress' in feature_name:
        return 'stress'
    else:
        return 'other'

importance_df = []
for (target, model_name), res in results.items():
    if model_name in ['XGBoost', 'RF']:
        model = res['model']
        if model_name == 'XGBoost':
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_
        for feat, imp in zip(feature_cols, importances):
            importance_df.append({
                'Target': target,
                'Model': model_name,
                'Feature': feat,
                'Importance': imp,
                'Category': categorize_feature(feat)
            })

imp_df = pd.DataFrame(importance_df)
# Save to Excel
imp_df.to_excel(OUTPUT_DIR / 'feature_importance.xlsx', index=False)

# Also save a summary of metrics
metrics_df = []
for (target, model), res in results.items():
    metrics_df.append({
        'Target': target,
        'Model': model,
        'R2': res['r2'],
        'MAE': res['mae'],
        'RMSE': res['rmse']
    })
metrics_df = pd.DataFrame(metrics_df)
metrics_df.to_excel(OUTPUT_DIR / 'metrics_summary.xlsx', index=False)

print("Hormone modeling complete.")