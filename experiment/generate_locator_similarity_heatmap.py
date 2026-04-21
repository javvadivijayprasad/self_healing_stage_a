import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

LOCATOR_FILE = REPORT_DIR / "locator_healing_report.csv"


def generate_heatmap():

    df = pd.read_csv(LOCATOR_FILE)

    df["original_locator"] = df["original_locator"].fillna("")
    df["healed_locator"] = df["healed_locator"].fillna("")

    original_locators = df["original_locator"].astype(str)
    healed_locators = df["healed_locator"].astype(str)

    locators = [l for l in list(original_locators) + list(healed_locators) if l.strip() != ""]

    if len(locators) < 2:
        print("Not enough locator data for similarity heatmap")
        return

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(locators)

    similarity_matrix = cosine_similarity(vectors)

    plt.figure(figsize=(10,8))
    sns.heatmap(similarity_matrix, cmap="coolwarm")

    plt.title("Locator Similarity Heatmap")

    plt.savefig(FIG_DIR / "locator_similarity_heatmap.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_heatmap()