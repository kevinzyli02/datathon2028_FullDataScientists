import pandas as pd
import json
from pathlib import Path
from typing import Set, Tuple
import logging
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)
logger = logging.getLogger(__name__)

def load_predefined_split(path: Path, sheet: str = "Summary") -> tuple[set[int], set[int]]:
    """
    Load train/test patient sets from Excel.

    Args:
        path: Path to Excel file.
        sheet: Sheet name containing 'Patient' and 'Test/Train' columns.

    Returns:
        train_patients, test_patients (disjoint sets of patient IDs).

    Raises:
        ValueError if columns missing, values invalid, or overlap.
    """
    print(f"DEBUG: Loading predefined split from: {path}")
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = df.columns.str.strip()
    required = {"Patient", "Test/Train"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required}")

    df = df.dropna(subset=["Patient"])
    df["Patient"] = df["Patient"].astype(int)
    df["Test/Train"] = df["Test/Train"].astype(str).str.lower().str.strip()
    valid = {"train", "test"}
    invalid = df[~df["Test/Train"].isin(valid)]
    if not invalid.empty:
        raise ValueError(f"Invalid Test/Train values: {invalid['Test/Train'].unique()}")

    train_patients = set(df[df["Test/Train"] == "train"]["Patient"])
    test_patients  = set(df[df["Test/Train"] == "test"]["Patient"])
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"Patients appear in both train and test: {overlap}")
    return train_patients, test_patients

def apply_split(df: pd.DataFrame, train_ids: Set[int], test_ids: Set[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by patient id.

    Args:
        df: DataFrame with 'id' column.
        train_ids: set of patient IDs for training.
        test_ids: set of patient IDs for testing.

    Returns:
        train_df, test_df.
    """
    train_df = df[df["id"].isin(train_ids)].copy()
    test_df  = df[df["id"].isin(test_ids)].copy()
    # Verify disjointness
    train_patients = set(train_df["id"].unique())
    test_patients  = set(test_df["id"].unique())
    if not train_patients.isdisjoint(test_patients):
        raise ValueError("Patient overlap detected after split!")
    return train_df, test_df

def write_split_manifest(
    train_ids: Set[int],
    test_ids: Set[int],
    df: pd.DataFrame,
    targets: list[str],
    output_dir: Path
):
    """Write a JSON manifest with patient coverage information."""
    present_ids = set(df["id"].unique())
    manifest = {
        "excel_train_patients": sorted(train_ids),
        "excel_test_patients": sorted(test_ids),
        "present_in_data_train": sorted(train_ids & present_ids),
        "present_in_data_test": sorted(test_ids & present_ids),
        "patients_with_no_rows": sorted((train_ids | test_ids) - present_ids),
        "labeled_rows": {}
    }
    for target in targets:
        if target not in df.columns:
            continue
        has_target = df[target].notna()
        manifest["labeled_rows"][target] = {
            "train": int(has_target[df["id"].isin(train_ids)].sum()),
            "test":  int(has_target[df["id"].isin(test_ids)].sum()),
        }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Split manifest written to {output_dir / 'split_manifest.json'}")