import argparse
from stage1_hormone import run_stage1
from stage2_cycle import run_stage2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hormone and/or cycle prediction pipeline.')
    parser.add_argument('--stage1', action='store_true', help='Run hormone prediction (Stage 1)')
    parser.add_argument('--stage2', action='store_true', help='Run cycle prediction (Stage 2)')
    parser.add_argument('--all', action='store_true', help='Run full pipeline (both stages)')
    args = parser.parse_args()

    if args.all or (not args.stage1 and not args.stage2):
        run_stage1()
        run_stage2()
    else:
        if args.stage1:
            run_stage1()
        if args.stage2:
            run_stage2()