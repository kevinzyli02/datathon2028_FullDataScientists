import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import config

# =============================================================================
# ENHANCEMENT 1: BASELINE NORMALIZATION
# =============================================================================
def normalize_by_baseline(df, targets, baseline_days=3):
    """Normalize hormone levels by patient-specific baseline."""
    df_norm = df.copy()
    for target in targets:
        if target not in df.columns:
            continue
        norm_col = f"{target}_normalized"
        df_norm[norm_col] = np.nan
        for pid in df['id'].unique():
            patient_data = df[df['id'] == pid].copy().sort_values('day_in_study')
            baseline = patient_data.head(baseline_days)[target].mean()
            if not np.isnan(baseline) and baseline > 0:
                mask = df_norm['id'] == pid
                df_norm.loc[mask, norm_col] = df_norm.loc[mask, target] / baseline
                print(f"   Patient {pid}: baseline {target} = {baseline:.2f}")
    return df_norm

# =============================================================================
# ENHANCEMENT 2: PERSONALIZED FEATURES
# =============================================================================
def add_personalized_features(df, targets, window_size=7):
    """Add patient-specific rolling features."""
    df_per = df.copy().sort_values(['id', 'day_in_study']).reset_index(drop=True)
    for target in targets:
        if target not in df.columns:
            continue
        print(f"   Adding personalized features for {target}...")
        for pid in df_per['id'].unique():
            mask = df_per['id'] == pid
            idx = df_per.index[mask]
            if len(idx) < 2:
                continue
            data = df_per.loc[idx, target].copy()
            # Rolling mean
            df_per.loc[idx, f'{target}_rolling_mean'] = data.rolling(window=min(window_size, len(data)), min_periods=1).mean()
            # Rolling std
            df_per.loc[idx, f'{target}_rolling_std'] = data.rolling(window=min(window_size, len(data)), min_periods=1).std().fillna(0)
            # Personal mean
            pmean = data.mean()
            df_per.loc[idx, f'{target}_personal_mean'] = pmean
            # Deviation
            df_per.loc[idx, f'{target}_deviation'] = data - pmean
            # Daily change
            df_per.loc[idx, f'{target}_daily_change'] = data.diff().fillna(0)
            # Cumulative
            df_per.loc[idx, f'{target}_cumulative'] = data.cumsum()
    if 'cycle_day' in df_per.columns:
        df_per['phase_follicular'] = (df_per['cycle_day'] <= 14).astype(int)
        df_per['phase_luteal'] = (df_per['cycle_day'] > 14).astype(int)
        df_per['phase_ovulation'] = ((df_per['cycle_day'] >= 12) & (df_per['cycle_day'] <= 16)).astype(int)
    return df_per

# =============================================================================
# FEATURE SELECTION / PROCESSING
# =============================================================================
def get_all_features(df, targets):
    """Get all available numeric features after cleaning, excluding target-related columns."""
    print("\n🎯 Processing all available features...")
    feature_sets = {}
    hormone_base = ['lh', 'estrogen', 'pdg']
    derived_patterns = ['_normalized', '_rolling_mean', '_rolling_std', '_personal_mean',
                        '_deviation', '_daily_change', '_cumulative']
    for target in targets:
        if target not in df.columns:
            continue
        print(f"\nProcessing features for {target}...")
        exclude_cols = targets + ['id', 'day_in_study', 'study_interval', 'is_weekend']
        for h in hormone_base:
            for pat in derived_patterns:
                col = f"{h}{pat}"
                if col in df.columns:
                    exclude_cols.append(col)
            if h != target.replace('_normalized','') and h in df.columns:
                exclude_cols.append(h)
        exclude_cols = list(set(exclude_cols))
        all_features = [c for c in df.columns if c not in exclude_cols]
        print(f"   Initial features: {len(all_features)}")
        numeric_features = [c for c in all_features if pd.api.types.is_numeric_dtype(df[c])]
        print(f"   Numeric features: {len(numeric_features)}")
        if len(numeric_features) == 0:
            feature_sets[target] = []
            continue
        # Remove constant columns
        constant = [c for c in numeric_features if df[c].nunique() <= 1]
        numeric_features = [c for c in numeric_features if c not in constant]
        print(f"   Removed {len(constant)} constant columns")
        # Remove high missingness (>80%)
        high_missing = [c for c in numeric_features if df[c].isnull().mean() > 0.8]
        numeric_features = [c for c in numeric_features if c not in high_missing]
        print(f"   Removed {len(high_missing)} high-missing columns")
        print(f"   ✅ Final features for {target}: {len(numeric_features)}")
        if len(numeric_features) > 0:
            cats = categorize_features(numeric_features)
            print("   Feature categories:")
            for cat, cnt in cats.items():
                print(f"     {cat}: {cnt}")
        feature_sets[target] = numeric_features
    return feature_sets

def categorize_features(features):
    cats = {
        'Sleep-related': sum('sleep' in f.lower() for f in features),
        'Stress-related': sum('stress' in f.lower() for f in features),
        'Heart-related': sum(('heart' in f.lower() or 'bpm' in f.lower()) for f in features),
        'Temperature-related': sum(('temp' in f.lower() or 'temperature' in f.lower()) for f in features),
        'Respiratory-related': sum(('respiratory' in f.lower() or 'breathing' in f.lower()) for f in features),
        'Exercise-related': sum(('exercise' in f.lower() or 'activity' in f.lower()) for f in features),
        'Glucose-related': sum('glucose' in f.lower() for f in features),
        'Symptom-related': sum(any(s in f.lower() for s in ['appetite','headache','cramp','breast','fatigue','mood','food','indigestion','bloating']) for f in features),
        'Other': 0
    }
    cats['Other'] = len(features) - sum(cats.values())
    return {k:v for k,v in cats.items() if v>0}

def get_cycle_features(df, include_predicted=True):
    """Return list of features for cycle prediction, optionally including predicted hormones."""
    hormone_base = ['lh', 'estrogen', 'pdg']
    derived_patterns = ['_normalized', '_rolling_mean', '_rolling_std', '_personal_mean',
                        '_deviation', '_daily_change', '_cumulative']
    exclude_cols = hormone_base + [f"{h}{p}" for h in hormone_base for p in derived_patterns] \
                   + ['id', 'day_in_study', 'study_interval', 'is_weekend'] + config.TARGETS
    exclude_cols += [f"{t}_normalized" for t in config.TARGETS if f"{t}_normalized" in df.columns]
    cycle_features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if include_predicted:
        pred_cols = [f"{h}_pred" for h in config.TARGETS if f"{h}_pred" in df.columns]
        cycle_features = list(set(cycle_features + pred_cols))
    return cycle_features

def select_top_features(X_train_scaled, y_train, feature_names, top_k=15):
    """Use RandomForest to select top_k features."""
    selector = RandomForestRegressor(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1)
    selector.fit(X_train_scaled, y_train)
    importances = selector.feature_importances_
    top_indices = np.argsort(importances)[-top_k:]
    top_features = [feature_names[i] for i in top_indices]
    print(f"Top {top_k} features: {top_features}")
    return top_indices, top_features