"""Runs the ranker ablation sweep end-to-end.

Executes run_experiment.py four times (heuristic, rule_based, random, none)
and then emits the mode-comparison table/figure. Results per mode are
preserved in reports/results_<mode>.csv; the heuristic run also updates
the canonical reports/results.csv so the rest of the analysis pipeline
keeps working.

Usage:
    python run_ablation_sweep.py                # defaults: 50 runs per mode
    python run_ablation_sweep.py --runs 10      # quick smoke sweep
    python run_ablation_sweep.py --modes heuristic rule_based
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODES = ["heuristic", "rule_based", "random", "none"]


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 60)
    print("$", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"command failed (exit={result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranker ablation sweep")
    parser.add_argument("--runs", type=int, default=50,
                        help="Experiment runs per mode (default 50).")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        help=f"Modes to sweep (default: {' '.join(DEFAULT_MODES)})")
    args = parser.parse_args()

    python = sys.executable
    started = time.time()

    for mode in args.modes:
        run([python, "run_experiment.py",
             "--healer-mode", mode,
             "--runs", str(args.runs)])

    run([python, "experiment/generate_mode_comparison.py"])

    elapsed = time.time() - started
    print(f"\nAblation sweep complete in {elapsed/60:.1f} min.")
    print("Artifacts:")
    print("  reports/mode_comparison.csv")
    print("  reports/mode_comparison_table.tex")
    print("  figures/mode_comparison.png")


if __name__ == "__main__":
    main()
