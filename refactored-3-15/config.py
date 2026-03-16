from dataclasses import dataclass
from pathlib import Path
import argparse

@dataclass
class Config:
    # Paths
    data_dir: Path = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data\processed data")
    output_dir: Path = Path("patient_wise_model_analysis")
    selfreport_path: Path = Path(r"C:\Users\kevin\PycharmProjects\datathon2028_FullDataScientists\data\McPhases SelfReport.xlsx")
    summary_sheet: str = "Summary"

    # Data settings
    sample_size: int = 15000
    test_size: float = 0.2          # not used when predefined split is used
    random_state: int = 42

    # Split and CV
    use_predefined_split: bool = True
    cv_folds: int = 3
    include_all_patients: bool = True

    # Feature engineering
    baseline_days: int = 3
    rolling_window: int = 7

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Enhanced hormone prediction pipeline")
        parser.add_argument("--use_predefined_split", type=lambda x: x.lower() == "true", default=True)
        parser.add_argument("--cv", type=int, default=3)
        # IMPORTANT: set default=None, not a string
        parser.add_argument("--selfreport_path", type=str, default=None,
                            help="Path to the Excel file with the predefined split")
        parser.add_argument("--summary_sheet", type=str, default="Summary")
        parser.add_argument("--include_all_patients", type=lambda x: x.lower() == "true", default=True)
        args = parser.parse_args()

        # Use dataclass default if argument not provided
        selfreport_path = Path(args.selfreport_path) if args.selfreport_path else cls.selfreport_path

        return cls(
            use_predefined_split=args.use_predefined_split,
            cv_folds=args.cv,
            selfreport_path=selfreport_path,
            summary_sheet=args.summary_sheet,
            include_all_patients=args.include_all_patients,
        )