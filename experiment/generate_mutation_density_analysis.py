"""Per-page mutation density vs. healing success rate.

Uses the existing mutation_report.csv (52 mutations over 5 HTML pages) and
results.csv (per-test outcomes) to correlate the density of mutations on a
page with the framework's healing performance on tests that touch that
page. No new experiment runs are required.

Outputs:
  - reports/mutation_density.csv
  - reports/mutation_density_table.tex
  - figures/mutation_density.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

# test_name -> primary page touched by the test. Kept minimal and auditable.
TEST_TO_PAGE = {
    "login_test":             "login.html",
    "inventory_test":         "inventory.html",
    "home_navigation_test":   "index.html",
    "cart_page_test":         "cart.html",
    "checkout_form_test":     "checkout.html",
    "products_listing_test":  "inventory.html",
    "multi_add_to_cart_test": "inventory.html",
    "navigation_flow_test":   "index.html",
}


def wilson_ci(successes: int, trials: int, z: float = 1.96):
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    muts = pd.read_csv(REPORT_DIR / "mutation_report.csv")
    res = pd.read_csv(REPORT_DIR / "results.csv")

    per_page = muts.groupby("page").size().rename("mutations").reset_index()
    res = res.copy()
    res["page"] = res["test_name"].map(TEST_TO_PAGE)

    rows = []
    for page, mut_count in per_page.itertuples(index=False):
        sub = res[res["page"] == page]
        healed = int(sub["healed_steps"].sum())
        failed = int(sub["failed_steps"].sum())
        broken = healed + failed
        if broken == 0:
            continue
        p, lo, hi = wilson_ci(healed, broken)
        rows.append({
            "page": page,
            "mutations": int(mut_count),
            "broken_events": broken,
            "healed": healed,
            "hsr": round(p, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })

    df = pd.DataFrame(rows).sort_values("mutations")
    df.to_csv(REPORT_DIR / "mutation_density.csv", index=False)
    print("Mutation-density table:")
    print(df.to_string(index=False))

    # Pearson correlation, inline implementation
    if len(df) >= 3:
        x = df["mutations"].to_numpy(dtype=float)
        y = df["hsr"].to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        num = ((x - xm) * (y - ym)).sum()
        den = math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
        pearson = num / den if den else float("nan")
    else:
        pearson = float("nan")

    # Figure
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df["mutations"], df["hsr"],
                yerr=[df["hsr"] - df["ci_lo"], df["ci_hi"] - df["hsr"]],
                fmt="o", capsize=4, color="#1f77b4")
    for _, r in df.iterrows():
        ax.annotate(r["page"], (r["mutations"], r["hsr"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mutations injected on page")
    ax.set_ylabel("Healing Success Rate")
    ax.set_ylim(0, 1.05)
    if not math.isnan(pearson):
        ax.set_title(f"HSR vs. Mutation Density (Pearson r = {pearson:.2f})")
    else:
        ax.set_title("HSR vs. Mutation Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mutation_density.png", dpi=300)
    plt.close()

    # LaTeX table
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Per-Page Mutation Density vs. Healing Success Rate (Wilson 95\% CI)}",
        r"\label{tab:mutation_density}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Page & Mutations & Broken & HSR & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        page = r["page"].replace("_", r"\_")
        lines.append(
            f"{page} & {int(r['mutations'])} & {int(r['broken_events'])} & "
            f"{r['hsr']*100:.1f}\\% & "
            f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    if not math.isnan(pearson):
        lines.append(
            rf"\vspace{{2pt}}\\\footnotesize Pearson $r={pearson:.2f}$ "
            r"between mutation count and HSR."
        )
    lines += [r"\end{table}"]
    (REPORT_DIR / "mutation_density_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\nArtifacts written.")


if __name__ == "__main__":
    main()
