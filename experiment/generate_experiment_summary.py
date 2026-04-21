from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_FILE = BASE_DIR / "reports" / "results.csv"
OUTPUT_CSV = BASE_DIR / "reports" / "experiment_summary.csv"
OUTPUT_TXT = BASE_DIR / "reports" / "experiment_summary.txt"


def main():

    print("Generating experiment summary...")

    df = pd.read_csv(RESULTS_FILE)

    total_runs = len(df)

    total_healed = df["healed_steps"].sum()
    total_failed = df["failed_steps"].sum()
    total_baseline = df["baseline_steps"].sum()

    broken_locators = total_healed + total_failed

    healing_success_rate = (
        total_healed / broken_locators
        if broken_locators else 0
    )

    locator_accuracy = (
        total_healed / broken_locators
        if broken_locators else 0
    )

    false_healing_rate = (
        total_failed / broken_locators
        if broken_locators else 0
    )

    summary = {
        "total_experiment_rows": total_runs,
        "baseline_steps": total_baseline,
        "broken_locators": broken_locators,
        "healed_locators": total_healed,
        "failed_healings": total_failed,
        "healing_success_rate": round(healing_success_rate, 4),
        "locator_recovery_accuracy": round(locator_accuracy, 4),
        "false_healing_rate": round(false_healing_rate, 4),
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_TXT, "w") as f:

        f.write("SELF-HEALING LOCATOR EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("Summary written to:")
    print(OUTPUT_CSV)
    print(OUTPUT_TXT)


if __name__ == "__main__":
    main()