from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments import run_config_directory, run_experiment_plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YAML experiment configs sequentially.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Folder containing .yaml/.yml experiment configs.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Single YAML experiment plan with defaults and experiments.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="CSV file where experiment rows are appended.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output CSV before running configs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-component progress logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_file is not None:
        rows = run_experiment_plan(
            plan_path=args.config_file,
            results_csv=args.results_csv,
            overwrite=args.overwrite,
            verbose=not args.quiet,
        )
    else:
        rows = run_config_directory(
            config_dir=args.config_dir,
            results_csv=args.results_csv,
            overwrite=args.overwrite,
            verbose=not args.quiet,
        )

    print()
    print(f"Finished {len(rows)} result rows")
    if args.results_csv is not None:
        print(f"Results CSV: {args.results_csv}")
    else:
        print("Results CSV: configured in YAML")


if __name__ == "__main__":
    main()
