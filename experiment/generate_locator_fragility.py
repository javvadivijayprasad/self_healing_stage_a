import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

RESULTS_FILE = REPORT_DIR / "locator_healing_report.csv"


def compute_fragility():

    df = pd.read_csv(RESULTS_FILE)

    grouped = df.groupby("element_name").agg(
        interactions=("element_name","count"),
        failures=("healing_failed","sum"),
        healed=("healing_success","sum")
    ).reset_index()

    grouped["fragility_score"] = grouped["failures"] / grouped["interactions"]

    top = grouped.sort_values(
        by="fragility_score",
        ascending=False
    ).head(10)

    rows = ""

    for _, r in top.iterrows():
        rows += f"{r['element_name']} & {r['interactions']} & {r['failures']} & {r['fragility_score']:.2f} \\\\\n"

    plt.figure()

    plt.barh(
        top["element_name"],
        top["fragility_score"]
    )

    plt.xlabel("Fragility Score")
    plt.title("Locator Fragility Analysis")

    plt.savefig(FIG_DIR / "locator_fragility.png", dpi=300)

    plt.close()


if __name__ == "__main__":
    compute_fragility()