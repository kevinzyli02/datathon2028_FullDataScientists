# period_model.py
"""
Train models to predict menstrual cycle phases:
- Regression: days until next fertility phase onset.
- Classification: is the day fertile?
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
DATA_DIR = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data\processed")
TRAIN_FILE = DATA_DIR / "train_data.parquet"
TEST_FILE = DATA_DIR / "test_data.parquet"
OUTPUT_DIR = DATA_DIR / "period_models"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------
# 1. Load data
train = pl.read_parquet(TRAIN_FILE).to_pandas()

# 2. Feature engineering (same as hormone model)
exclude_cols = ['id', 'day_in_study', 'phase', 'study_interval', 'lh', 'estrogen', 'pdg']  # exclude hormones too
feature_cols = [c for c in train.columns if c not in exclude_cols and train[c].dtype in ['float64', 'int64']]
print(f"Feature columns: {feature_cols}")

# Sort by id and day
train = train.sort_values(['id', 'day_in_study'])
test = test.sort_values(['id', 'day_in_study'])

# Forward fill missing values per patient
for col in feature_cols:
    train[col] = train.groupby('id')[col].fillna(method='ffill')
    test[col] = test.groupby('id')[col].fillna(method='ffill')
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# Scale features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[feature_cols].values)
X_test_scaled = scaler.transform(test[feature_cols].values)

# 3. Create regression target: days until next fertility onset
def days_to_next_fertility(df):
    """Add a column 'days_to_fertility' for each row: days until the next 'Fertility' phase.
       If no future fertility, set to NaN."""
    df = df.copy()
    df['days_to_fertility'] = np.nan
    for pid, group in df.groupby('id'):
        group = group.sort_values('day_in_study')
        # Find indices where phase is Fertility
        fert_idx = group[group['phase'] == 'Fertility'].index
        # For each day, find the next fertility day
        for idx in group.index:
            future_fert = fert_idx[fert_idx > idx]
            if len(future_fert) > 0:
                next_fert_day = group.loc[future_fert[0], 'day_in_study']
                days = next_fert_day - group.loc[idx, 'day_in_study']
                df.loc[idx, 'days_to_fertility'] = days
    return df

train = days_to_next_fertility(train)
test = days_to_next_fertility(test)

# Drop rows where target is NaN (no future fertility)
train_reg = train.dropna(subset=['days_to_fertility'])
test_reg = test.dropna(subset=['days_to_fertility'])

# 4. Create classification target: fertile (1 if phase == 'Fertility', else 0)
train['is_fertile'] = (train['phase'] == 'Fertility').astype(int)
test['is_fertile'] = (test['phase'] == 'Fertility').astype(int)

# For classification, we can use all rows (including non-fertile)
X_train_clf = train[feature_cols].values
X_test_clf = test[feature_cols].values
y_train_clf = train['is_fertile'].values
y_test_clf = test['is_fertile'].values

# For regression, use only rows with target
X_train_reg = train_reg[feature_cols].values
y_train_reg = train_reg['days_to_fertility'].values
X_test_reg = test_reg[feature_cols].values
y_test_reg = test_reg['days_to_fertility'].values

# Scale for NN
X_train_reg_scaled = scaler.transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)
X_train_clf_scaled = scaler.transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# 5. Regression models (XGBoost, Random Forest, LSTM/GRU)
def evaluate_regression(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    # PAE3: percentage of predictions within 3 days
    pae3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
    print(f"{name}: MAE={mae:.4f}, PAE3={pae3:.2f}%")
    return mae, pae3

reg_results = {}

# XGBoost
xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train_reg, y_train_reg)
pred_xgb = xgb_reg.predict(X_test_reg)
mae, pae3 = evaluate_regression(y_test_reg, pred_xgb, "XGBoost")
reg_results['XGBoost'] = {'mae': mae, 'pae3': pae3, 'model': xgb_reg, 'pred': pred_xgb}

# Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
pred_rf = rf_reg.predict(X_test_reg)
mae, pae3 = evaluate_regression(y_test_reg, pred_rf, "Random Forest")
reg_results['RF'] = {'mae': mae, 'pae3': pae3, 'model': rf_reg, 'pred': pred_rf}

# LSTM/GRU - need sequences
def create_sequences_reg(X_df, y_df, seq_len=7):
    X_seq, y_seq = [], []
    for pid, group in X_df.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        y_p = y_df.loc[group.index].values
        for i in range(len(X_p) - seq_len):
            X_seq.append(X_p[i:i+seq_len])
            y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# For regression, we need to use only rows where target exists; but sequences must be contiguous.
# We'll create sequences from the original train_reg, but we need to ensure that each sequence
# has valid target at the end. We'll create sequences from the entire train set and then filter.
# Simpler: we can create sequences from train_reg (which already has target) but we must preserve patient order.
# However, train_reg may have gaps. To avoid complexity, we'll create sequences from the original train,
# then use only those sequences where the last day has a valid target.
# We'll scale features first.
train_scaled = scaler.transform(train[feature_cols].values)
train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols)
train_scaled_df['id'] = train['id'].values
train_scaled_df['day_in_study'] = train['day_in_study'].values
train_scaled_df['days_to_fertility'] = train['days_to_fertility'].values

def create_sequences_reg_scaled(df, seq_len=7):
    X_seq, y_seq = [], []
    for pid, group in df.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        y_p = group['days_to_fertility'].values
        for i in range(len(X_p) - seq_len):
            if not np.isnan(y_p[i+seq_len]):  # only keep if target is valid
                X_seq.append(X_p[i:i+seq_len])
                y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_train_seq_reg, y_train_seq_reg = create_sequences_reg_scaled(train_scaled_df, seq_len=7)
# For test, similar
test_scaled = scaler.transform(test[feature_cols].values)
test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols)
test_scaled_df['id'] = test['id'].values
test_scaled_df['day_in_study'] = test['day_in_study'].values
test_scaled_df['days_to_fertility'] = test['days_to_fertility'].values
X_test_seq_reg, y_test_seq_reg = create_sequences_reg_scaled(test_scaled_df, seq_len=7)

print(f"Regression sequences: Train {X_train_seq_reg.shape}, Test {X_test_seq_reg.shape}")

# Build LSTM and GRU for regression
def build_reg_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

input_shape = (7, len(feature_cols))
lstm_reg = build_reg_lstm(input_shape)
gru_reg = Sequential([
    GRU(64, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])
gru_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

lstm_reg.fit(X_train_seq_reg, y_train_seq_reg, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_lstm = lstm_reg.predict(X_test_seq_reg).flatten()
mae, pae3 = evaluate_regression(y_test_seq_reg, pred_lstm, "LSTM")
reg_results['LSTM'] = {'mae': mae, 'pae3': pae3, 'model': lstm_reg, 'pred': pred_lstm}

gru_reg.fit(X_train_seq_reg, y_train_seq_reg, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_gru = gru_reg.predict(X_test_seq_reg).flatten()
mae, pae3 = evaluate_regression(y_test_seq_reg, pred_gru, "GRU")
reg_results['GRU'] = {'mae': mae, 'pae3': pae3, 'model': gru_reg, 'pred': pred_gru}

# 6. Classification models (XGBoost, Random Forest, LSTM/GRU)
def evaluate_classification(y_true, y_pred_prob, name):
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"{name}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
    return acc, auc

clf_results = {}

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_clf.fit(X_train_clf, y_train_clf)
pred_prob_xgb = xgb_clf.predict_proba(X_test_clf)[:, 1]
acc, auc = evaluate_classification(y_test_clf, pred_prob_xgb, "XGBoost")
clf_results['XGBoost'] = {'acc': acc, 'auc': auc, 'model': xgb_clf, 'pred_prob': pred_prob_xgb}

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
pred_prob_rf = rf_clf.predict_proba(X_test_clf)[:, 1]
acc, auc = evaluate_classification(y_test_clf, pred_prob_rf, "Random Forest")
clf_results['RF'] = {'acc': acc, 'auc': auc, 'model': rf_clf, 'pred_prob': pred_prob_rf}

# LSTM/GRU for classification (binary)
def build_clf_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create sequences for classification (use all days)
def create_sequences_clf(df, seq_len=7):
    X_seq, y_seq = [], []
    for pid, group in df.groupby('id'):
        group = group.sort_values('day_in_study')
        X_p = group[feature_cols].values
        y_p = group['is_fertile'].values
        for i in range(len(X_p) - seq_len):
            X_seq.append(X_p[i:i+seq_len])
            y_seq.append(y_p[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# Scale features
train_clf_scaled = scaler.transform(train[feature_cols].values)
train_clf_scaled_df = pd.DataFrame(train_clf_scaled, columns=feature_cols)
train_clf_scaled_df['id'] = train['id'].values
train_clf_scaled_df['day_in_study'] = train['day_in_study'].values
train_clf_scaled_df['is_fertile'] = train['is_fertile'].values

test_clf_scaled = scaler.transform(test[feature_cols].values)
test_clf_scaled_df = pd.DataFrame(test_clf_scaled, columns=feature_cols)
test_clf_scaled_df['id'] = test['id'].values
test_clf_scaled_df['day_in_study'] = test['day_in_study'].values
test_clf_scaled_df['is_fertile'] = test['is_fertile'].values

X_train_seq_clf, y_train_seq_clf = create_sequences_clf(train_clf_scaled_df, seq_len=7)
X_test_seq_clf, y_test_seq_clf = create_sequences_clf(test_clf_scaled_df, seq_len=7)

print(f"Classification sequences: Train {X_train_seq_clf.shape}, Test {X_test_seq_clf.shape}")

lstm_clf = build_clf_lstm(input_shape)
gru_clf = Sequential([
    GRU(64, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
gru_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_clf.fit(X_train_seq_clf, y_train_seq_clf, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_prob_lstm = lstm_clf.predict(X_test_seq_clf).flatten()
acc, auc = evaluate_classification(y_test_seq_clf, pred_prob_lstm, "LSTM")
clf_results['LSTM'] = {'acc': acc, 'auc': auc, 'model': lstm_clf, 'pred_prob': pred_prob_lstm}

gru_clf.fit(X_train_seq_clf, y_train_seq_clf, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
pred_prob_gru = gru_clf.predict(X_test_seq_clf).flatten()
acc, auc = evaluate_classification(y_test_seq_clf, pred_prob_gru, "GRU")
clf_results['GRU'] = {'acc': acc, 'auc': auc, 'model': gru_clf, 'pred_prob': pred_prob_gru}

# 7. Save results
reg_metrics = pd.DataFrame([{'Model': k, **v} for k, v in reg_results.items()])
reg_metrics.to_excel(OUTPUT_DIR / 'regression_metrics.xlsx', index=False)

clf_metrics = pd.DataFrame([{'Model': k, **v} for k, v in clf_results.items()])
clf_metrics.to_excel(OUTPUT_DIR / 'classification_metrics.xlsx', index=False)

# 8. Visualizations
# Regression: scatter plot of actual vs predicted for best model (by MAE)
best_reg_model = min(reg_results.items(), key=lambda x: x[1]['mae'])[0]
pred_best = reg_results[best_reg_model]['pred']
plt.figure(figsize=(6,6))
plt.scatter(y_test_seq_reg, pred_best, alpha=0.5)
plt.plot([y_test_seq_reg.min(), y_test_seq_reg.max()], [y_test_seq_reg.min(), y_test_seq_reg.max()], 'r--')
plt.xlabel('Actual days to fertility')
plt.ylabel('Predicted days to fertility')
plt.title(f'Best Regression Model: {best_reg_model} (MAE={reg_results[best_reg_model]["mae"]:.2f})')
plt.savefig(OUTPUT_DIR / 'regression_best_scatter.png')
plt.show()

# Classification: ROC curves for all models
plt.figure()
for model_name, res in clf_results.items():
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test_seq_clf if model_name in ['LSTM','GRU'] else y_test_clf, res['pred_prob'])
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={res["auc"]:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Fertility Classification')
plt.legend()
plt.savefig(OUTPUT_DIR / 'classification_roc.png')
plt.show()

print("Period modeling complete.")