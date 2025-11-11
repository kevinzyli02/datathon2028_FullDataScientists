# run_complete_analysis.py
"""
MASTER SCRIPT: Runs preprocessing + feature selection sequentially
"""

import subprocess
import sys
import os
from pathlib import Path


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run([sys.executable, script_name],
                                capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    """Run the complete analysis pipeline"""
    print("🚀 STARTING COMPLETE ANALYSIS PIPELINE")

    # Step 1: Run preprocessing
    success1 = run_script("robust_sequential_analysis.py", "Data Preprocessing & Merging")

    if not success1:
        print("❌ Preprocessing failed. Cannot continue with feature selection.")
        return

    # Step 2: Run feature selection
    success2 = run_script("feature_selection_standalone.py", "Feature Selection")

    if success1 and success2:
        print(f"\n🎉 PIPELINE COMPLETE!")
        print("📁 Check these directories for results:")
        print("   - robust_sequential_analysis/ → Preprocessed data")
        print("   - feature_selection_results/ → Selected features")
    else:
        print(f"\n⚠️  Pipeline completed with errors")


if __name__ == "__main__":
    main()