"""Ablation comparison across healer modes.

Reads every `reports/results_<mode>.csv` present (plus the canonical
`reports/results.csv`, treated as the heuristic-mode run) and emits a
side-by-side comparison of healing success rate, false heal rate, and
Wilson 95% confidence intervals. Used to answer: "how much of the
framework's recovery rate is due to similarity scoring vs. a rule-based
attribute fallback vs. random chance?"

Missing modes are skipped silently so the script can be run at any point
during an incremental ablation sweep.

Outputs:
  - reports/mode_comparison.csv
  - reports/mode_comparison_table.tex
  - figures/mode_comparison.png
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

MODES = ["heuristic", "rule_based", "random", "ml", "hybrid", "none"]


def wilson_ci(successes: int, trials: int, z: float = 1.96):
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_mode(mode: str) -> pd.DataFrame | None:
    if mode == "heuristic":
        path = REPORT_DIR / "results.csv"
    else:
        path = REPORT_DIR / f"results_{mode}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # If the file was produced by an older run_experiment without a mode
    # column, tag it so downstream groupby is consistent.
    if "healer_mode" not in df.columns:
        df["healer_mode"] = mode
    return df


def main() -> None:
    rows = []
    loaded = {}
    for mode in MODES:
        df = load_mode(mode)
        if df is None or df.empty:
            continue
        loaded[mode] = df
        healed = int(df["healed_steps"].sum())
        failed = int(df["failed_steps"].sum())
        broken = healed + failed
        p, lo, hi = wilson_ci(healed, broken)
        rows.append({
            "mode": mode,
            "healed": healed,
            "failed": failed,
            "broken": broken,
            "hsr": round(p, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "fhr": round(1 - p, 4) if broken else 0.0,
        })

    if not rows:
        print("[mode_comparison] no results files found; run run_experiment.py "
              "with --healer-mode to populate reports/results_<mode>.csv.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "mode_comparison.csv", index=False)
    print("Mode comparison:")
    print(out.to_string(index=False))

    # LaTeX
    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation: Healing Success Rate by Ranker Mode (Wilson 95\% CI)}",
        r"\label{tab:mode_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Mode & Healed & Broken & HSR & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        nm = r["mode"].replace("_", r"\_")
        latex.append(
            f"{nm} & {int(r['healed'])} & {int(r['broken'])} & "
            f"{r['hsr']*100:.1f}\\% & "
            f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORT_DIR / "mode_comparison_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    # Figure
    FIG_DIR.mkdir(exist_ok=True)
    x = np.arange(len(out))
    p = out["hsr"].to_numpy()
    lo = out["ci_lo"].to_numpy()
    hi = out["ci_hi"].to_numpy()
    err = np.vstack([p - lo, hi - p])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f"]
    ax.bar(x, p, yerr=err, capsize=4,
           color=colors[:len(out)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(out["mode"], rotation=15, ha="right")
    ax.set_ylabel("Healing Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ranker Ablation: HSR with Wilson 95% CI")
    for i, v in enumerate(p):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mode_comparison.png", dpi=300)
    plt.close()

    print("\nArtifacts written to reports/mode_comparison.* and figures/mode_comparison.png")


if __name__ == "__main__":
    main()
