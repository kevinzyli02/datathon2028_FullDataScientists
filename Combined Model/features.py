import polars as pl
import numpy as np
import config

def normalize_by_baseline_polars(df, targets, baseline_days=3):
    """Normalize hormones by first `baseline_days` per patient."""
    df = df.sort(['id', 'day_in_study'])
    for target in targets:
        if target not in df.columns:
            continue
        # Compute baseline value per patient (mean of first baseline_days)
        baseline = df.group_by('id').agg(
            pl.col(target).head(baseline_days).mean().alias(f'{target}_baseline')
        )
        df = df.join(baseline, on='id', how='left')
        df = df.with_columns(
            (pl.col(target) / pl.col(f'{target}_baseline')).alias(f'{target}_normalized')
        )
        # Drop baseline column
        df = df.drop(f'{target}_baseline')
        # Print some baselines for verification
        print(f"   Added {target}_normalized")
    return df

def add_personalized_features_polars(df, targets, window_size=7):
    """Add rolling statistics per patient."""
    df = df.sort(['id', 'day_in_study'])
    for target in targets:
        if target not in df.columns:
            continue
        print(f"   Adding personalized features for {target}...")
        # Rolling mean and std per patient (use group_by + rolling)
        df = df.with_columns([
            pl.col(target).rolling_mean(window_size, min_periods=1).over('id').alias(f'{target}_rolling_mean'),
            pl.col(target).rolling_std(window_size, min_periods=1).over('id').alias(f'{target}_rolling_std'),
        ])
        # Fill NaN in rolling_std with 0
        df = df.with_columns(pl.col(f'{target}_rolling_std').fill_null(0))

        # Personal mean (overall) per patient
        personal_mean = df.group_by('id').agg(pl.col(target).mean().alias(f'{target}_personal_mean'))
        df = df.join(personal_mean, on='id', how='left')
        # Deviation
        df = df.with_columns(
            (pl.col(target) - pl.col(f'{target}_personal_mean')).alias(f'{target}_deviation')
        )
        # Daily change (diff)
        df = df.with_columns(
            pl.col(target).diff().over('id').alias(f'{target}_daily_change')
        ).with_columns(pl.col(f'{target}_daily_change').fill_null(0))

        # Cumulative sum
        df = df.with_columns(
            pl.col(target).cum_sum().over('id').alias(f'{target}_cumulative')
        )

    # Add cycle phases if cycle_day exists
    if 'cycle_day' in df.columns:
        df = df.with_columns([
            (pl.col('cycle_day') <= 14).cast(pl.Int32).alias('phase_follicular'),
            (pl.col('cycle_day') > 14).cast(pl.Int32).alias('phase_luteal'),
            ((pl.col('cycle_day') >= 12) & (pl.col('cycle_day') <= 16)).cast(pl.Int32).alias('phase_ovulation')
        ])
    return df

def get_all_features_polars(df, targets):
    """Return dictionary of feature lists per target (Polars version)."""
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
        numeric_features = [c for c in all_features if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        print(f"   Numeric features: {len(numeric_features)}")
        if len(numeric_features) == 0:
            feature_sets[target] = []
            continue
        # Remove constant columns (unique count = 1)
        constant = []
        for col in numeric_features:
            if df[col].n_unique() <= 1:
                constant.append(col)
        numeric_features = [c for c in numeric_features if c not in constant]
        print(f"   Removed {len(constant)} constant columns")
        # Remove high missing (>80%)
        high_missing = []
        for col in numeric_features:
            null_count = df[col].null_count()
            if null_count / df.height > 0.8:
                high_missing.append(col)
        numeric_features = [c for c in numeric_features if c not in high_missing]
        print(f"   Removed {len(high_missing)} high-missing columns")
        print(f"   ✅ Final features for {target}: {len(numeric_features)}")
        if len(numeric_features) > 0:
            cats = categorize_features_polars(numeric_features)
            print("   Feature categories:")
            for cat, cnt in cats.items():
                print(f"     {cat}: {cnt}")
        feature_sets[target] = numeric_features
    return feature_sets

def categorize_features_polars(features):
    # Same as before, but using list comprehension (can reuse the same function, it's just string matching)
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

def get_cycle_features_polars(df, include_predicted=True):
    """Return list of features for cycle prediction (Polars version)."""
    hormone_base = ['lh', 'estrogen', 'pdg']
    derived_patterns = ['_normalized', '_rolling_mean', '_rolling_std', '_personal_mean',
                        '_deviation', '_daily_change', '_cumulative']
    exclude_cols = hormone_base + [f"{h}{p}" for h in hormone_base for p in derived_patterns] \
                   + ['id', 'day_in_study', 'study_interval', 'is_weekend'] + config.TARGETS
    exclude_cols += [f"{t}_normalized" for t in config.TARGETS if f"{t}_normalized" in df.columns]
    cycle_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    if include_predicted:
        pred_cols = [f"{h}_pred" for h in config.TARGETS if f"{h}_pred" in df.columns]
        cycle_features = list(set(cycle_features + pred_cols))
    return cycle_features