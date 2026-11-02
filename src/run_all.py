"""Run the whole project from start to finish (PR 9, integration).

Runs the same commands as the "How to run" steps in README.md, in order,
and stops at the first step that fails.

Run from the project folder (after downloading the datasets):
    python -m src.run_all              # cleaning, models, tuning, efficiency, errors (about 50 minutes)
    python -m src.run_all --test       # final test-set scores (run once, at the end)
    python -m src.run_all --dry-run    # only print the commands

The default run uses the training data only. The test set is read only
with --test (team rule: once, for the final results), after the default
run has saved the models.
"""

import argparse
import subprocess
import sys

# Step name and the module arguments, in the order they must run.
TRAIN_STEPS = [
    ("Clean the data and split it", ["src.prepare_data"]),
    ("Baseline models, raw columns", ["src.models"]),
    ("Baseline models, with features", ["src.models", "--tag", "features"]),
    ("Tune the models", ["src.tuning"]),
    ("Measure memory and runtime", ["src.efficiency"]),
    ("Analyse the errors", ["src.error_analysis"]),
]

# Final scores on the test set. Each step loads models saved by TRAIN_STEPS.
TEST_STEPS = [
    ("Test baseline models, raw columns", ["src.models", "--test"]),
    ("Test baseline models, with features", ["src.models", "--tag", "features", "--test"]),
    ("Test tuned models", ["src.tuning", "--test"]),
    ("Test-set errors", ["src.error_analysis", "--test"]),
]


def run_steps(steps, dry_run=False):
    """Run each step as `python -m ...`; stop if one of them fails."""
    for number, (name, args) in enumerate(steps, start=1):
        # sys.executable is the Python that runs this script (the .venv one)
        command = [sys.executable, "-m", *args]
        print(f"\n=== Step {number}/{len(steps)}: {name}")
        print("    python -m " + " ".join(args))
        if not dry_run:
            subprocess.run(command, check=True)   # check=True stops on an error


def main():
    """Read the options and run the training steps or the test steps."""
    parser = argparse.ArgumentParser(description="Run the whole project in order.")
    parser.add_argument("--test", action="store_true",
                        help="run only the final test-set steps (once, at the end)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    args = parser.parse_args()
    run_steps(TEST_STEPS if args.test else TRAIN_STEPS, args.dry_run)
    print("\nDone. Results are in outputs/results/ and charts in outputs/figures/.")


if __name__ == "__main__":
    main()
