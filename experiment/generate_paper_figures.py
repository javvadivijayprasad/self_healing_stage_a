import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_FILE = BASE_DIR / "reports" / "results.csv"
MUTATION_FILE = BASE_DIR / "reports" / "mutation_report.csv"

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def load_results():
    return pd.read_csv(RESULTS_FILE)


# -------------------------------------------------
# Healing Success Rate
# -------------------------------------------------

def figure_healing_success_rate(df):

    healed = df["healed_steps"].sum()
    failed = df["failed_steps"].sum()

    rate = healed / (healed + failed) if (healed + failed) else 0

    plt.figure()

    plt.bar(["Healing Success Rate"], [rate])

    plt.ylim(0,1)

    plt.ylabel("Rate")

    plt.title("Self-Healing Success Rate")

    plt.savefig(FIGURE_DIR / "healing_success_rate.png")

    plt.close()


# -------------------------------------------------
# Locator Recovery Accuracy
# -------------------------------------------------

def figure_locator_recovery_accuracy(df):

    healed = df["healed_steps"].sum()
    failed = df["failed_steps"].sum()

    accuracy = healed / (healed + failed) if (healed + failed) else 0

    plt.figure()

    plt.bar(["Locator Recovery Accuracy"], [accuracy])

    plt.ylim(0,1)

    plt.title("Locator Recovery Accuracy")

    plt.savefig(FIGURE_DIR / "locator_recovery_accuracy.png")

    plt.close()


# -------------------------------------------------
# False Healing Rate
# -------------------------------------------------

def figure_false_healing_rate(df):

    healed = df["healed_steps"].sum()
    failed = df["failed_steps"].sum()

    false_rate = failed / (healed + failed) if (healed + failed) else 0

    plt.figure()

    plt.bar(["False Healing Rate"], [false_rate])

    plt.ylim(0,1)

    plt.title("False Healing Rate")

    plt.savefig(FIGURE_DIR / "false_healing_rate.png")

    plt.close()


# -------------------------------------------------
# Version Comparison
# -------------------------------------------------

def figure_version_comparison(df):

    grouped = df.groupby("ui_version").sum(numeric_only=True)

    healed = grouped["healed_steps"]
    failed = grouped["failed_steps"]

    success_rate = healed / (healed + failed)

    plt.figure()

    success_rate.plot(kind="bar")

    plt.ylabel("Success Rate")

    plt.title("Healing Performance Across UI Versions")

    plt.savefig(FIGURE_DIR / "version_comparison.png")

    plt.close()


# -------------------------------------------------
# Healing Distribution
# -------------------------------------------------

def figure_healing_distribution(df):

    totals = [
        df["baseline_steps"].sum(),
        df["healed_steps"].sum(),
        df["failed_steps"].sum()
    ]

    labels = [
        "Baseline Steps",
        "Healed Steps",
        "Failed Steps"
    ]

    plt.figure()

    plt.pie(
        totals,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title("Distribution of Locator Outcomes")

    plt.savefig(FIGURE_DIR / "healing_distribution.png")

    plt.close()


# -------------------------------------------------
# Execution Improvement
# -------------------------------------------------

def figure_execution_improvement(df):

    baseline = df["baseline_steps"].sum()
    healed = df["healed_steps"].sum()

    values = [baseline, healed]

    labels = ["Baseline Execution", "Recovered by Healing"]

    plt.figure()

    plt.bar(labels, values)

    plt.ylabel("Steps")

    plt.title("Execution Improvement with Self-Healing")

    plt.savefig(FIGURE_DIR / "execution_improvement.png")

    plt.close()


# -------------------------------------------------
# Cumulative Healing
# -------------------------------------------------

def figure_cumulative_healing(df):

    df = df.copy()

    df["cumulative_healed"] = df["healed_steps"].cumsum()

    plt.figure()

    plt.plot(df.index, df["cumulative_healed"])

    plt.xlabel("Test Run")

    plt.ylabel("Cumulative Healed Steps")

    plt.title("Cumulative Healing Performance")

    plt.savefig(FIGURE_DIR / "cumulative_healing.png")

    plt.close()


# -------------------------------------------------
# Mutation vs Healing
# -------------------------------------------------

def figure_mutations_vs_healing():

    if not MUTATION_FILE.exists():
        print("mutation_report.csv not found — skipping mutation figure")
        return

    mutation_df = pd.read_csv(MUTATION_FILE)

    if "mutations" not in mutation_df.columns:
        print("mutations column missing — skipping figure")
        return

    results_df = load_results()

    healed = results_df["healed_steps"].sum()
    failed = results_df["failed_steps"].sum()

    success_rate = healed / (healed + failed)

    versions = mutation_df["ui_version"]
    mutations = mutation_df["mutations"]

    healing_rates = [success_rate for _ in range(len(mutations))]

    plt.figure()

    plt.scatter(mutations, healing_rates)

    plt.xlabel("Number of DOM Mutations")

    plt.ylabel("Healing Success Rate")

    plt.title("DOM Mutations vs Healing Success")

    plt.savefig(FIGURE_DIR / "mutation_vs_healing.png")

    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    df = load_results()

    print("Generating evaluation figures...")

    figure_healing_success_rate(df)

    figure_locator_recovery_accuracy(df)

    figure_false_healing_rate(df)

    figure_version_comparison(df)

    figure_healing_distribution(df)

    figure_execution_improvement(df)

    figure_cumulative_healing(df)

    figure_mutations_vs_healing()

    print("Figures saved in /figures directory")


if __name__ == "__main__":
    main()