from pathlib import Path
import argparse

from subspace_face_recognition.experiments import run_experiment_plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YAML experiment plan sequentially.",
    )
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        default=Path("configs/experiment_plan.yaml"),
        help="YAML experiment plan with defaults and an experiments list.",
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
        help="Delete the output CSV before running the plan.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-component progress logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_experiment_plan(
        plan_path=args.plan,
        results_csv=args.results_csv,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

    print()
    print(f"Finished {len(result.rows)} result rows")
    print(f"Results CSV: {result.results_csv}")


if __name__ == "__main__":
    main()
