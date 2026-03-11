import pandas as pd
import numpy as np
from pathlib import Path
import os
import gc
from sklearn.model_selection import train_test_split
import config

def preprocess_data(df):
    """Preprocess data with symptom mapping and memory optimization."""
    # Aggressive numeric conversion
    for col in df.columns:
        if col == 'id':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(np.int32)
            continue
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Symptom mapping
    symptom_mapping = {
        'Not at all': 0, 'Very Low/Little': 1, 'Very Low': 1, 'Low': 2,
        'Moderate': 3, 'High': 4, 'Very High': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }
    symptom_cols = [
        'appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts',
        'fatigue', 'sleepissue', 'moodswing', 'stress', 'foodcravings',
        'indigestion', 'bloating'
    ]
    for col in symptom_cols:
        if col in df.columns:
            df[col] = df[col].map(symptom_mapping)

    # Downcast to save memory
    fcols = df.select_dtypes('float').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    icols = df.select_dtypes('integer').columns
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df

def handle_file_structure(df, file_name):
    """Rename columns and aggregate high-frequency files."""
    if file_name in ['sleep.csv', 'computed_temperature.csv']:
        if 'sleep_start_day_in_study' in df.columns:
            df = df.rename(columns={'sleep_start_day_in_study': 'day_in_study'})
    elif file_name == 'exercise.csv':
        if 'start_day_in_study' in df.columns:
            df = df.rename(columns={'start_day_in_study': 'day_in_study'})

    if file_name in ['glucose.csv', 'wrist_temperature.csv', 'daily_norm_unmodified.csv']:
        print(f"   📊 Aggregating {file_name} by participant...")
        df = aggregate_by_participant(df)
    return df

def aggregate_by_participant(df):
    if 'id' not in df.columns:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'id']
    if numeric_cols:
        agg_df = df.groupby('id')[numeric_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
        agg_df.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in agg_df.columns]
        print(f"   ✅ Aggregated to {agg_df.shape}")
        return agg_df
    return df

def find_merge_keys(base_df, new_df, file_name):
    common_cols = set(base_df.columns) & set(new_df.columns)
    if 'id' in common_cols and 'day_in_study' in common_cols:
        return ['id', 'day_in_study']
    elif 'id' in common_cols:
        return ['id']
    else:
        day_columns = ['sleep_start_day_in_study', 'start_day_in_study', 'day_in_study']
        for day_col in day_columns:
            if day_col in new_df.columns and 'day_in_study' in base_df.columns:
                new_df.rename(columns={day_col: 'day_in_study'}, inplace=True)
                return ['id', 'day_in_study']
    return None

def load_comprehensive_data(files, data_dir, sample_size=config.SAMPLE_SIZE):
    """Load and merge all data files with memory efficiency."""
    print("📂 Loading comprehensive dataset with memory optimization...")
    base_file = files[0]
    df = pd.read_csv(data_dir / base_file)
    df = preprocess_data(df)
    print(f"Base data: {df.shape} | Memory: {df.memory_usage().sum() / 1e6:.2f} MB")

    for file_name in files[1:]:
        print(f"🔗 Merging {file_name}...")
        try:
            file_path = data_dir / file_name
            if not file_path.exists():
                continue
            # Load only necessary columns
            temp_head = pd.read_csv(file_path, nrows=0)
            cols_to_load = [c for c in temp_head.columns if not any(x in c.lower() for x in ['comment','note','desc','source'])]
            new_data = pd.read_csv(file_path, usecols=cols_to_load)
            new_data = preprocess_data(new_data)
            new_data = handle_file_structure(new_data, file_name)

            merge_keys = find_merge_keys(df, new_data, file_name)
            if not merge_keys:
                del new_data
                continue

            if len(new_data) > 40000:
                new_data = new_data.sample(n=30000, random_state=config.RANDOM_STATE)
                print(f"   🔽 Pre-sampled {file_name} to {new_data.shape}")

            df = pd.merge(df, new_data, on=merge_keys, how='left',
                          suffixes=('', f'_{file_name.replace(".csv", "")}'))
            print(f"✅ Merged {file_name}. Current shape: {df.shape}")

            del new_data
            gc.collect()
        except Exception as e:
            print(f"❌ Error with {file_name}: {e}")
            gc.collect()

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=config.RANDOM_STATE)
    df = preprocess_data(df)  # final downcast
    return df

def create_consistency_report(main_file_path, data_dir, file_list):
    """Check patient overlap between main file and others."""
    print("\n📋 CREATING DATA CONSISTENCY REPORT")
    print("=" * 50)
    main_df = pd.read_csv(main_file_path)
    main_patients = set(main_df['id'].unique())
    report_data = []
    for file_name in file_list:
        file_path = data_dir / file_name
        if not file_path.exists():
            report_data.append({'file': file_name, 'status': 'MISSING', 'patients': 0,
                                'overlap_with_main': 0, 'overlap_percent': 0, 'records': 0})
            continue
        df = pd.read_csv(file_path)
        records = len(df)
        if 'id' in df.columns:
            patients = set(df['id'].unique())
            overlap = patients & main_patients
            overlap_percent = len(overlap) / len(main_patients) * 100 if main_patients else 0
            report_data.append({'file': file_name, 'status': 'OK', 'patients': len(patients),
                                'overlap_with_main': len(overlap), 'overlap_percent': overlap_percent,
                                'records': records})
        else:
            report_data.append({'file': file_name, 'status': 'NO_ID_COLUMN', 'patients': 'N/A',
                                'overlap_with_main': 'N/A', 'overlap_percent': 'N/A', 'records': records})
    report_df = pd.DataFrame(report_data)
    print("\n📊 DATA CONSISTENCY REPORT:")
    print(report_df.to_string(index=False))
    files_to_filter = [item['file'] for item in report_data if item['status'] == 'OK' and item['overlap_percent'] < 100]
    if files_to_filter:
        print(f"\n🎯 FILES THAT NEED FILTERING: {len(files_to_filter)}")
        for f in files_to_filter:
            print(f"   - {f}")
    else:
        print("\n✅ All files already consistent with main file!")
    return report_df

def filter_all_files_to_match_patients(main_file_path, data_dir, output_dir, file_list):
    """Filter each file to only patients present in main file."""
    print("🔍 FILTERING DATA FILES TO MATCH PATIENTS IN MAIN FILE")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    main_df = pd.read_csv(main_file_path)
    valid_patient_ids = set(main_df['id'].unique())
    print(f"✅ Found {len(valid_patient_ids)} valid patients in main file")
    for file_name in file_list:
        print(f"\n🔄 Processing {file_name}...")
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"   ⚠️ File not found: {file_name}")
            continue
        df = pd.read_csv(file_path)
        orig_size = len(df)
        orig_patients = len(df['id'].unique()) if 'id' in df.columns else "No ID column"
        print(f"   Original: {orig_size} records, {orig_patients} patients")
        if 'id' in df.columns:
            filtered_df = df[df['id'].isin(valid_patient_ids)].copy()
            filt_size = len(filtered_df)
            filt_patients = len(filtered_df['id'].unique())
            print(f"   Filtered: {filt_size} records, {filt_patients} patients")
            print(f"   Removed: {orig_size - filt_size} records ({((orig_size - filt_size)/orig_size*100):.1f}%)")
            out_path = output_dir / file_name
            filtered_df.to_csv(out_path, index=False)
            print(f"   💾 Saved filtered file to: {out_path}")
        else:
            print(f"   ⚠️ No 'id' column, copying as-is")
            out_path = output_dir / file_name
            df.to_csv(out_path, index=False)
            print(f"   💾 Copied file to: {out_path}")
    print(f"\n✅ FILTERING COMPLETE! Files saved to {output_dir}")
    return output_dir

def save_patient_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, output_dir=config.OUTPUT_DIR):
    """Perform a single patient-wise split and save train/test patient IDs."""
    all_patients = df['id'].unique()
    train_patients, test_patients = train_test_split(all_patients, test_size=test_size, random_state=random_state)
    split_df = pd.DataFrame({
        'patient_id': np.concatenate([train_patients, test_patients]),
        'set': ['train'] * len(train_patients) + ['test'] * len(test_patients)
    })
    split_df.to_csv(output_dir / 'patient_split.csv', index=False)
    print(f"✅ Patient split saved to {output_dir / 'patient_split.csv'}")
    return train_patients, test_patients

def load_patient_split(output_dir=config.OUTPUT_DIR):
    split_df = pd.read_csv(output_dir / 'patient_split.csv')
    train_patients = split_df[split_df['set'] == 'train']['patient_id'].values
    test_patients = split_df[split_df['set'] == 'test']['patient_id'].values
    return train_patients, test_patients