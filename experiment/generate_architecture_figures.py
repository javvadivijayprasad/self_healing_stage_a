from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[1]

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def draw_pipeline_diagram(title, steps, output_file):

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.6 + 1)

    ax.axis("off")

    x_center = 5

    y_positions = list(
        reversed([1 + i * 1.6 for i in range(len(steps))])
    )

    for i, (step, y) in enumerate(zip(steps, y_positions)):

        width = 6.6
        height = 0.9

        left = x_center - width / 2
        bottom = y - height / 2

        box = FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3,
            edgecolor="black",
            facecolor="white"
        )

        ax.add_patch(box)

        ax.text(
            x_center,
            y,
            step,
            ha="center",
            va="center",
            fontsize=11,
            wrap=True
        )

        if i < len(steps) - 1:

            next_y = y_positions[i + 1]

            ax.annotate(
                "",
                xy=(x_center, next_y + 0.45),
                xytext=(x_center, y - 0.45),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.4
                )
            )

    ax.set_title(title, fontsize=14, pad=18)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# -----------------------------------
# Framework Architecture
# -----------------------------------

def generate_framework_architecture():

    steps = [

        "Test Script",

        "Test Execution Engine (Selenium)",

        "Locator Failure Detection",

        "DOM Snapshot Capture",

        "Candidate Element Extraction",

        "Similarity Feature Generation",

        "Heuristic Ranking",

        "Recovered Locator Selection",

        "Test Execution Continues"

    ]

    draw_pipeline_diagram(

        title="AI-Driven Self-Healing Test Automation Framework",

        steps=steps,

        output_file="framework_architecture_diagram.png"

    )


# -----------------------------------
# Experiment Pipeline
# -----------------------------------

def generate_experiment_pipeline():

    steps = [

        "Baseline Web Application (version_1)",

        "DOM Mutation Generator",

        "Mutated Application Versions (version_2 – version_5)",

        "Bootstrap Locator Metadata Capture",

        "Selenium Test Execution",

        "Self-Healing Locator Recovery",

        "Experiment Result Logging (results.csv)",

        "Locator Healing Report Generation",

        "Experiment Analysis & Figure Generation",

        "Paper-Ready Evaluation Outputs"

    ]

    draw_pipeline_diagram(

        title="End-to-End Experimental Pipeline",

        steps=steps,

        output_file="experiment_pipeline_diagram.png"

    )


# -----------------------------------
# Main
# -----------------------------------

def main():

    print("Generating architecture figures...")

    generate_framework_architecture()

    generate_experiment_pipeline()

    print("Figures generated in /figures directory")


if __name__ == "__main__":
    main()