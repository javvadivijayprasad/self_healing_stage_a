"""
Threshold sweep and denominator analysis for the self-healing framework.

Two analyses are produced:

1. THRESHOLD SWEEP. The heuristic acceptance threshold is varied from 0.50
   to 0.85 in steps of 0.05. For each threshold T, we reclassify every
   heal_success whose healing_score < T as a failure. This yields a
   {T -> HSR_T, retained_success_count, conservative_FHR_T} curve that
   shows the tunable operating point of the framework without re-running
   the experiment. Note: this is a CONSERVATIVE reclassification. It gives
   a lower bound on HSR at each threshold, because the experiment selected
   candidates with the current threshold; a live run at a higher threshold
   might sometimes have selected a different (possibly lower-scored)
   candidate. The conservative curve is nonetheless valuable because it
   shows that the framework could trade a modest HSR drop for a higher
   lower-bound on recovery confidence.

2. DENOMINATOR DECOMPOSITION. We decompose the 850 unresolved failures
   into two classes based on the healer error messages:
   (a) "ranked-but-unresolved" - a candidate was scored and selected but
       could not be resolved in the live DOM (the candidate was hidden,
       detached, or the CSS-path hint was invalid at click time), and
   (b) "no-candidate-ranked" - the candidate extractor returned zero
       candidates or no candidate crossed the threshold.
   The distinction matters because the two classes suggest different
   engineering fixes: ranked-but-unresolved points to recovery-executor
   robustness (stale elements, visibility, waits), while no-candidate
   points to extractor/scoring coverage (feature engineering, ML fallback).

Outputs:
  - reports/threshold_sweep.csv
  - reports/denominator_decomposition.csv
  - reports/threshold_sweep_table.tex
  - figures/threshold_sweep.png
  - figures/denominator_decomposition.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

HEALING_FILE = REPORT_DIR / "locator_healing_report.csv"
RESULTS_FILE = REPORT_DIR / "results.csv"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def threshold_sweep(healing: pd.DataFrame) -> pd.DataFrame:
    attempts = healing[healing["status"].isin(["heal_success", "heal_failed"])].copy()
    total_attempts = len(attempts)

    # Original heal_success scores
    healed = attempts[attempts["status"] == "heal_success"].copy()
    healed_scores = pd.to_numeric(healed["healing_score"], errors="coerce").fillna(0.0)

    rows = []
    for t in THRESHOLDS:
        retained = int((healed_scores >= t).sum())
        demoted = int(len(healed_scores)) - retained
        # At this threshold, retained are healings; everything else is a failure
        failures = demoted + (len(attempts) - len(healed))
        hsr = retained / total_attempts if total_attempts else 0.0
        fhr = failures / total_attempts if total_attempts else 0.0
        rows.append({
            "threshold": t,
            "retained_heals": retained,
            "demoted_heals": demoted,
            "failures": failures,
            "attempts": total_attempts,
            "hsr": round(hsr, 4),
            "fhr": round(fhr, 4),
        })
    return pd.DataFrame(rows)


def denominator_decomposition(results: pd.DataFrame, healing: pd.DataFrame):
    """Split the 850 failures into ranked-but-unresolved vs no-candidate."""
    err = results[results["status"] == "error"].copy()
    details = err["details"].fillna("").astype(str)

    ranked_unresolved = details.str.contains(
        "Healing ranked a candidate but could not resolve it", regex=False
    ).sum()
    no_candidate = len(err) - int(ranked_unresolved)

    # Healed events (where a candidate was both ranked AND resolved)
    healed_events = int((healing["status"] == "heal_success").sum())
    # Total broken events = healed + ranked_unresolved + no_candidate
    total_broken = healed_events + int(ranked_unresolved) + no_candidate

    rows = [
        {"category": "ranked_and_resolved (healed)", "count": healed_events},
        {"category": "ranked_but_unresolved", "count": int(ranked_unresolved)},
        {"category": "no_candidate_ranked", "count": no_candidate},
    ]
    df = pd.DataFrame(rows)
    df["share_of_broken"] = (df["count"] / total_broken).round(4)

    return df, total_broken


def main():
    healing = pd.read_csv(HEALING_FILE)
    results = pd.read_csv(RESULTS_FILE)

    sweep = threshold_sweep(healing)
    sweep.to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)
    print("Threshold sweep:")
    print(sweep.to_string(index=False))

    decomp, total_broken = denominator_decomposition(results, healing)
    decomp.to_csv(REPORT_DIR / "denominator_decomposition.csv", index=False)
    print("\nDenominator decomposition (total broken =", total_broken, "):")
    print(decomp.to_string(index=False))

    # ---------------- Threshold sweep figure ----------------
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sweep["threshold"], sweep["hsr"], marker="o", color="#2ca02c",
             label="HSR (lower-bound)")
    ax1.plot(sweep["threshold"], sweep["fhr"], marker="s", color="#d62728",
             label="FHR (upper-bound)")
    ax1.set_xlabel("Heuristic acceptance threshold")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Threshold Sweep: HSR and FHR vs Acceptance Threshold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right")
    for _, r in sweep.iterrows():
        ax1.annotate(f"{r['hsr']*100:.0f}%",
                     xy=(r["threshold"], r["hsr"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_sweep.png", dpi=300)
    plt.close()

    # ---------------- Decomposition figure ----------------
    fig, ax2 = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax2.bar(decomp["category"], decomp["count"], color=colors,
            edgecolor="black")
    ax2.set_ylabel("Events")
    ax2.set_title("Decomposition of Broken-Locator Events")
    for i, (c, s) in enumerate(zip(decomp["count"], decomp["share_of_broken"])):
        ax2.text(i, c + max(decomp["count"]) * 0.01,
                 f"{int(c)}\n({s*100:.1f}%)", ha="center", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "denominator_decomposition.png", dpi=300)
    plt.close()

    # ---------------- LaTeX table ----------------
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Threshold Sweep: Conservative HSR and FHR as a Function of Acceptance Threshold}",
        r"\label{tab:threshold_sweep}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Threshold & Retained & Demoted & Failures & HSR & FHR \\",
        r"\midrule",
    ]
    for _, r in sweep.iterrows():
        lines.append(
            f"{r['threshold']:.2f} & "
            f"{int(r['retained_heals'])} & "
            f"{int(r['demoted_heals'])} & "
            f"{int(r['failures'])} & "
            f"{r['hsr']*100:.1f}\\% & "
            f"{r['fhr']*100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORT_DIR / "threshold_sweep_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\nArtifacts written.")


if __name__ == "__main__":
    main()
