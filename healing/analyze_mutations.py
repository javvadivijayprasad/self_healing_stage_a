from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Reports folder
REPORT_DIR = BASE_DIR / "reports"

# Mutation report file
MUTATION_FILE = REPORT_DIR / "mutation_report.csv"


def main():

    df = pd.read_csv(MUTATION_FILE)

    print("\nTotal mutations:", len(df))

    print("\nMutation types:")
    print(df["mutation_type"].value_counts())

    print("\nPages affected:")
    print(df["page"].value_counts())


    # -----------------------------------
    # Create mutation distribution chart
    # -----------------------------------

    page_counts = df["page"].value_counts()

    plt.figure(figsize=(6,4))

    page_counts.plot(kind="bar")

    plt.title("DOM Mutations per Page")

    plt.ylabel("Mutation Count")

    plt.xlabel("Page")

    plt.tight_layout()

    output = REPORT_DIR / "mutations_per_page.png"

    plt.savefig(output)

    print("\nChart saved:", output)


if __name__ == "__main__":
    main()