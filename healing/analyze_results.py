import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
RESULT_FILE = REPORT_DIR / "results.csv"


def load_results():

    df = pd.read_csv(RESULT_FILE)

    print("Columns detected:", df.columns.tolist())

    return df


def compute_metrics(df):

    total_runs = len(df)

    success_runs = df["success"].sum()

    success_rate = success_runs / total_runs

    healed_total = df["healed_steps"].sum()

    failed_total = df["failed_steps"].sum()

    return {
        "total_runs": total_runs,
        "success_runs": success_runs,
        "success_rate": success_rate,
        "healed_steps": healed_total,
        "failed_steps": failed_total
    }


def plot_success_rate(metrics):

    plt.figure(figsize=(6,4))

    labels = ["Test Success Rate"]
    values = [metrics["success_rate"]]

    plt.bar(labels, values)

    plt.title("Self-Healing Locator Success Rate")
    plt.ylabel("Success Rate")
    plt.ylim(0,1)

    output = REPORT_DIR / "healing_success_rate.png"

    plt.savefig(output)

    print("Saved:", output)


def plot_healing_effect(metrics):

    plt.figure(figsize=(6,4))

    labels = ["Healed Steps", "Failed Steps"]

    values = [
        metrics["healed_steps"],
        metrics["failed_steps"]
    ]

    plt.bar(labels, values)

    plt.title("Locator Healing vs Failure")

    output = REPORT_DIR / "healing_vs_failure.png"

    plt.savefig(output)

    print("Saved:", output)


def generate_summary(metrics):

    output = REPORT_DIR / "experiment_summary.txt"

    with open(output, "w") as f:

        f.write("Self-Healing Locator Experiment Summary\n")
        f.write("--------------------------------------\n\n")

        f.write(f"Total Runs: {metrics['total_runs']}\n")

        f.write(f"Successful Runs: {metrics['success_runs']}\n")

        f.write(f"Success Rate: {metrics['success_rate']:.2f}\n\n")

        f.write(f"Healed Steps: {metrics['healed_steps']}\n")

        f.write(f"Failed Steps: {metrics['failed_steps']}\n")

    print("Saved:", output)


def main():

    print("Analyzing experiment results...")

    df = load_results()

    metrics = compute_metrics(df)

    plot_success_rate(metrics)

    plot_healing_effect(metrics)

    generate_summary(metrics)

    print("\nAnalysis Complete")


if __name__ == "__main__":
    main()