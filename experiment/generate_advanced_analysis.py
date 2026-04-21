import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

RESULTS_FILE = REPORT_DIR / "results.csv"
MUTATION_FILE = REPORT_DIR / "mutation_report.csv"


def top_failed_tests(results):

    df = results[["test_name", "failed_steps"]].sort_values(
        by="failed_steps", ascending=False
    )

    top = df.head(10)

    plt.figure()
    plt.barh(top["test_name"], top["failed_steps"])
    plt.title("Top Tests With Locator Failures")
    plt.xlabel("Failed Steps")

    plt.savefig(FIG_DIR / "top_failed_tests.png", dpi=300)
    plt.close()

    rows = ""
    for _, r in top.iterrows():
        rows += f"{r['test_name']} & {r['failed_steps']} \\\\\n"

def mutation_impact(mutations):
    """Plot healed vs failed healing attempts per mutation class.

    Previous version plotted only the injection count, which did not show
    any healing "impact". The richer view uses the mutation_healing_summary
    produced by generate_mutation_vs_healing_table.py so the figure
    actually conveys recovery behavior per mutation class.

    The LaTeX table is intentionally NOT written here — the richer
    per-class recovery table is emitted by generate_mutation_vs_healing_table.py
    so that both files share a single source of truth.
    """
    summary_path = REPORT_DIR / "mutation_healing_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary.sort_values("attempts", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(summary["mutation_type"], summary["healed"],
                color="#2ca02c", label="Healed")
        ax.barh(summary["mutation_type"], summary["failures"],
                left=summary["healed"], color="#d62728", label="Failed")
        ax.set_xlabel("Healing attempts")
        ax.set_title("Per-Mutation-Class Healing Outcomes")
        ax.legend(loc="lower right")
        for i, (h, f_, rate) in enumerate(zip(
                summary["healed"], summary["failures"], summary["recovery_rate"])):
            total = h + f_
            if total > 0:
                ax.text(total + max(summary["attempts"]) * 0.01, i,
                        f"{rate*100:.0f}%", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "mutation_failure_impact.png", dpi=300)
        plt.close()
    else:
        # Fallback: plot injection counts only
        counts = mutations["mutation_type"].value_counts()
        plt.figure()
        counts.plot(kind="bar")
        plt.title("Mutation Type Distribution")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "mutation_failure_impact.png", dpi=300)
        plt.close()


def healing_efficiency(results):

    df = results.copy()

    df["baseline_steps"] = df["baseline_steps"].replace(0, 1)
    df["healing_efficiency"] = df["healed_steps"] / df["baseline_steps"]
    df["healing_efficiency"] = df["healing_efficiency"].fillna(0)

    top = df.sort_values(by="healing_efficiency", ascending=False).head(10)

    plt.figure()
    plt.barh(top["test_name"], top["healing_efficiency"])
    plt.title("Healing Efficiency per Test")

    plt.savefig(FIG_DIR / "healing_efficiency.png", dpi=300)
    plt.close()

    rows = ""
    for _, r in top.iterrows():
        rows += f"{r['test_name']} & {r['healing_efficiency']:.2f} \\\\\n"

    latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Healing Efficiency per Test}}
\\begin{{tabular}}{{l c}}
\\toprule
Test Name & Healing Efficiency \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with open(REPORT_DIR / "healing_efficiency_table.tex", "w") as f:
        f.write(latex)


def main():

    results = pd.read_csv(RESULTS_FILE)
    mutations = pd.read_csv(MUTATION_FILE)

    top_failed_tests(results)
    mutation_impact(mutations)
    healing_efficiency(results)

    print("Advanced experiment analysis generated.")


if __name__ == "__main__":
    main()