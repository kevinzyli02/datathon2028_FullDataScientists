import polars as pl
from pathlib import Path
import config

def convert_csv_to_parquet(data_dir=config.DATA_DIR, parquet_dir=config.PARQUET_DIR):
    parquet_dir.mkdir(exist_ok=True)
    for file in config.COMPREHENSIVE_FILES:
        csv_path = data_dir / file
        if not csv_path.exists():
            print(f"⚠️ {file} not found, skipping.")
            continue
        parquet_path = parquet_dir / file.replace('.csv', '.parquet')
        print(f"Converting {file} to Parquet...")
        # Read with Polars (lazy, but we'll collect for conversion)
        df = pl.read_csv(csv_path, ignore_errors=True)
        df.write_parquet(parquet_path)
        print(f"✅ Saved to {parquet_path}")

if __name__ == "__main__":
    convert_csv_to_parquet()