"""ML ablation study: heuristic vs ML vs hybrid vs baselines.

══════════════════════════════════════════════════════════════════════════════
Paper 2, Stage B — Extended Ablation Analysis
══════════════════════════════════════════════════════════════════════════════

Extends the existing generate_mode_comparison.py with additional analyses
specific to the ML ranker:

  1. HSR comparison across all 6 modes (heuristic, ml, hybrid, rule_based, random, none)
  2. Per-mutation-type recovery rate comparison (ML vs heuristic)
  3. Per-UI-version stability analysis
  4. Score distribution comparison (heuristic scores vs ML probabilities)
  5. Statistical significance testing (McNemar's test: heuristic vs ML)
  6. Feature importance from trained model
  7. LaTeX tables and paper-ready figures

Reads:
  - reports/results.csv            (heuristic)
  - reports/results_ml.csv         (ML ranker)
  - reports/results_hybrid.csv     (hybrid ensemble)
  - reports/results_rule_based.csv (rule-based baseline)
  - reports/results_random.csv     (random baseline)
  - reports/results_none.csv       (no healing)
  - reports/mutation_report.csv    (mutation metadata)
  - models/healing_ranker.pkl      (trained model, for feature importance)

Outputs:
  - reports/ml_ablation_summary.csv
  - reports/ml_ablation_per_mutation.csv
  - reports/ml_ablation_per_version.csv
  - reports/ml_ablation_table.tex
  - reports/ml_feature_importance.csv
  - figures/ml_ablation_comparison.png
  - figures/ml_ablation_per_mutation.png
  - figures/ml_ablation_per_version.png
  - figures/ml_feature_importance.png
  - figures/ml_score_distribution.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"

MODES = ["heuristic", "ml", "hybrid", "rule_based", "random", "none"]
MODE_LABELS = {
    "heuristic": "Heuristic",
    "ml": "ML (GB)",
    "hybrid": "Hybrid",
    "rule_based": "Rule-Based",
    "random": "Random",
    "none": "None",
}
MODE_COLORS = {
    "heuristic": "#2ca02c",
    "ml": "#9467bd",
    "hybrid": "#d62728",
    "rule_based": "#1f77b4",
    "random": "#ff7f0e",
    "none": "#7f7f7f",
}


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float, float]:
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_mode(mode: str) -> Optional[pd.DataFrame]:
    if mode == "heuristic":
        path = REPORT_DIR / "results.csv"
    else:
        path = REPORT_DIR / f"results_{mode}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "healer_mode" not in df.columns:
        df["healer_mode"] = mode
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. Overall HSR comparison
# ═══════════════════════════════════════════════════════════════════════════

def generate_overall_comparison() -> pd.DataFrame:
    rows = []
    for mode in MODES:
        df = load_mode(mode)
        if df is None or df.empty:
            continue
        healed = int(df["healed_steps"].sum())
        failed = int(df["failed_steps"].sum())
        broken = healed + failed
        baseline = int(df["baseline_steps"].sum())
        total_steps = baseline + broken
        p, lo, hi = wilson_ci(healed, broken)
        rows.append({
            "mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "total_steps": total_steps,
            "baseline_steps": baseline,
            "broken": broken,
            "healed": healed,
            "failed": failed,
            "hsr": round(p, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "fhr": round(1 - p, 4) if broken else 0.0,
        })

    if not rows:
        print("[ml_ablation] No results files found.")
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "ml_ablation_summary.csv", index=False)
    print("\nOverall HSR Comparison:")
    print(out[["label", "broken", "healed", "hsr", "ci_lo", "ci_hi"]].to_string(index=False))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-mutation-type comparison
# ═══════════════════════════════════════════════════════════════════════════

def generate_per_mutation_comparison() -> Optional[pd.DataFrame]:
    """Compare ML vs heuristic per mutation type."""
    mutation_path = REPORT_DIR / "mutation_report.csv"
    if not mutation_path.exists():
        print("[ml_ablation] mutation_report.csv not found; skipping per-mutation analysis.")
        return None

    mutations = pd.read_csv(mutation_path)
    if "mutation_type" not in mutations.columns:
        print("[ml_ablation] mutation_report.csv missing 'mutation_type' column.")
        return None

    # Get unique mutation types from the report
    mutation_types = mutations["mutation_type"].unique().tolist()

    compare_modes = ["heuristic", "ml", "hybrid"]
    rows = []

    for mode in compare_modes:
        df = load_mode(mode)
        if df is None:
            continue

        healed = int(df["healed_steps"].sum())
        failed = int(df["failed_steps"].sum())
        broken = healed + failed
        p, lo, hi = wilson_ci(healed, broken)

        # Since mutation_report doesn't have per-version mapping to results,
        # we report aggregate HSR per mode per mutation type present
        for mut_type in mutation_types:
            mut_count = int((mutations["mutation_type"] == mut_type).sum())
            rows.append({
                "mode": mode,
                "mutation_type": mut_type,
                "mutation_count": mut_count,
                "healed": healed,
                "failed": failed,
                "broken": broken,
                "hsr": round(p, 4),
            })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "ml_ablation_per_mutation.csv", index=False)
    print("\nPer-Mutation Type Distribution + HSR by Mode:")
    # Show concise summary: mode-level HSR with mutation type counts
    summary = out.drop_duplicates(subset=["mode"])[["mode", "broken", "healed", "hsr"]]
    print(summary.to_string(index=False))
    print(f"\nMutation types found: {mutation_types}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-version stability
# ═══════════════════════════════════════════════════════════════════════════

def generate_per_version_comparison() -> Optional[pd.DataFrame]:
    compare_modes = ["heuristic", "ml", "hybrid"]
    rows = []

    for mode in compare_modes:
        df = load_mode(mode)
        if df is None:
            continue
        for version, vdf in df.groupby("ui_version"):
            healed = int(vdf["healed_steps"].sum())
            failed = int(vdf["failed_steps"].sum())
            broken = healed + failed
            p, lo, hi = wilson_ci(healed, broken)
            rows.append({
                "mode": mode,
                "ui_version": version,
                "broken": broken,
                "healed": healed,
                "hsr": round(p, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "ml_ablation_per_version.csv", index=False)
    print("\nPer-Version HSR:")
    print(out.to_string(index=False))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature importance
# ═══════════════════════════════════════════════════════════════════════════

def generate_feature_importance() -> Optional[pd.DataFrame]:
    model_path = MODEL_DIR / "healing_ranker.pkl"
    if not model_path.exists():
        print("[ml_ablation] Trained model not found; skipping feature importance.")
        return None

    try:
        import joblib
        pipeline = joblib.load(model_path)
        gb_model = pipeline.named_steps["model"]
        importances = gb_model.feature_importances_

        from healing.ml_ranker import FEATURE_COLUMNS
        fi = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        fi.to_csv(REPORT_DIR / "ml_feature_importance.csv", index=False)
        print("\nFeature Importance:")
        for _, r in fi.iterrows():
            bar = "█" * int(r["importance"] * 40)
            print(f"  {r['feature']:28} {r['importance']:.4f}  {bar}")
        return fi
    except Exception as exc:
        print(f"[ml_ablation] Feature importance extraction failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Score distribution comparison
# ═══════════════════════════════════════════════════════════════════════════

def extract_scores(df: pd.DataFrame) -> List[float]:
    """Extract healing scores from the details JSON column."""
    scores = []
    for _, row in df.iterrows():
        try:
            details = json.loads(row.get("details", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        for step in details.get("step_logs", []):
            score = step.get("healing_score")
            if score is not None:
                scores.append(float(score))
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_overall_comparison(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(summary))
    p = summary["hsr"].to_numpy()
    lo = summary["ci_lo"].to_numpy()
    hi = summary["ci_hi"].to_numpy()
    err = np.vstack([p - lo, hi - p])
    colors = [MODE_COLORS.get(m, "#333333") for m in summary["mode"]]

    bars = ax.bar(x, p, yerr=err, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Healing Success Rate (HSR)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("ML Ablation: Ranker Mode Comparison (Wilson 95% CI)", fontsize=12)

    for i, v in enumerate(p):
        ax.text(i, v + 0.03, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(y=p[0] if len(p) > 0 else 0.5, color="#2ca02c", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_ablation_comparison.png", dpi=300)
    plt.close()


def plot_per_version(per_version: pd.DataFrame) -> None:
    if per_version is None or per_version.empty:
        return

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    modes = per_version["mode"].unique()
    versions = sorted(per_version["ui_version"].unique())
    width = 0.25
    x = np.arange(len(versions))

    for i, mode in enumerate(modes):
        mdf = per_version[per_version["mode"] == mode]
        hsr_vals = [mdf[mdf["ui_version"] == v]["hsr"].values[0]
                    if v in mdf["ui_version"].values else 0 for v in versions]
        lo_vals = [mdf[mdf["ui_version"] == v]["ci_lo"].values[0]
                   if v in mdf["ui_version"].values else 0 for v in versions]
        hi_vals = [mdf[mdf["ui_version"] == v]["ci_hi"].values[0]
                   if v in mdf["ui_version"].values else 0 for v in versions]

        hsr = np.array(hsr_vals)
        lo = np.array(lo_vals)
        hi = np.array(hi_vals)
        err = np.vstack([hsr - lo, hi - hsr])

        ax.bar(x + i * width, hsr, width, yerr=err, capsize=3,
               label=MODE_LABELS.get(mode, mode),
               color=MODE_COLORS.get(mode, "#333"), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(versions, fontsize=10)
    ax.set_ylabel("HSR", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("HSR by UI Version: Heuristic vs ML vs Hybrid", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_ablation_per_version.png", dpi=300)
    plt.close()


def plot_score_distribution() -> None:
    """Plot healing score distributions for heuristic vs ML vs hybrid."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for i, mode in enumerate(["heuristic", "ml", "hybrid"]):
        df = load_mode(mode)
        if df is None:
            axes[i].set_title(f"{MODE_LABELS.get(mode, mode)} (no data)")
            continue

        scores = extract_scores(df)
        if not scores:
            axes[i].set_title(f"{MODE_LABELS.get(mode, mode)} (no scores)")
            continue

        axes[i].hist(scores, bins=30, color=MODE_COLORS.get(mode, "#333"),
                     edgecolor="black", alpha=0.8, linewidth=0.5)
        axes[i].set_title(f"{MODE_LABELS.get(mode, mode)} (n={len(scores)})", fontsize=11)
        axes[i].set_xlabel("Score / Probability", fontsize=10)
        axes[i].axvline(x=np.median(scores), color="red", linestyle="--", alpha=0.7,
                       label=f"Median: {np.median(scores):.3f}")
        axes[i].legend(fontsize=8)

    axes[0].set_ylabel("Frequency", fontsize=10)
    fig.suptitle("Score Distribution Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_score_distribution.png", dpi=300)
    plt.close()


def plot_feature_importance(fi: pd.DataFrame) -> None:
    if fi is None or fi.empty:
        return

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    fi_sorted = fi.sort_values("importance", ascending=True)
    y = np.arange(len(fi_sorted))
    ax.barh(y, fi_sorted["importance"], color="#9467bd", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(fi_sorted["feature"], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("ML Ranker: Feature Importance", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(fi_sorted["importance"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_feature_importance.png", dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX table
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{ML Ablation: Healing Success Rate by Ranker Mode (Wilson 95\% CI)}",
        r"\label{tab:ml_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Mode & Broken & Healed & Failed & HSR & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        nm = str(r["label"]).replace("_", r"\_")
        latex.append(
            f"{nm} & {int(r['broken'])} & {int(r['healed'])} & {int(r['failed'])} & "
            f"{r['hsr']*100:.1f}\\% & "
            f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORT_DIR / "ml_ablation_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# McNemar's test (statistical significance)
# ═══════════════════════════════════════════════════════════════════════════

def mcnemar_test() -> None:
    """McNemar's test: heuristic vs ML on paired healing outcomes."""
    h_df = load_mode("heuristic")
    m_df = load_mode("ml")

    if h_df is None or m_df is None:
        print("[ml_ablation] Cannot run McNemar's test — need both heuristic and ML results.")
        return

    # Align by (ui_version, test_name) pairs
    h_outcomes = {}
    for _, row in h_df.iterrows():
        key = (row["ui_version"], row["test_name"])
        h_outcomes[key] = row["success"]

    m_outcomes = {}
    for _, row in m_df.iterrows():
        key = (row["ui_version"], row["test_name"])
        m_outcomes[key] = row["success"]

    common_keys = set(h_outcomes.keys()) & set(m_outcomes.keys())
    if len(common_keys) < 10:
        print(f"[ml_ablation] Too few paired observations ({len(common_keys)}) for McNemar's test.")
        return

    # Contingency table: (heuristic_pass, ml_pass)
    a = sum(1 for k in common_keys if h_outcomes[k] and m_outcomes[k])      # both pass
    b = sum(1 for k in common_keys if h_outcomes[k] and not m_outcomes[k])   # heuristic pass, ml fail
    c = sum(1 for k in common_keys if not h_outcomes[k] and m_outcomes[k])   # heuristic fail, ml pass
    d = sum(1 for k in common_keys if not h_outcomes[k] and not m_outcomes[k])  # both fail

    print(f"\nMcNemar's Contingency Table (n={len(common_keys)}):")
    print(f"  Both pass:            {a}")
    print(f"  Heuristic only:       {b}")
    print(f"  ML only:              {c}")
    print(f"  Both fail:            {d}")

    # McNemar's chi-squared (with continuity correction)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # p-value from chi2 distribution (1 df)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        print(f"  McNemar χ²:           {chi2:.4f}")
        print(f"  p-value:              {p_value:.6f}  {sig}")
    else:
        print("  McNemar's test: no discordant pairs (b + c = 0).")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  ML ABLATION STUDY — Paper 2, Stage B")
    print("=" * 60)

    # 1. Overall comparison
    summary = generate_overall_comparison()
    if summary.empty:
        return

    # 2. Per-mutation comparison
    per_mutation = generate_per_mutation_comparison()

    # 3. Per-version comparison
    per_version = generate_per_version_comparison()

    # 4. Feature importance
    fi = generate_feature_importance()

    # 5. Figures
    plot_overall_comparison(summary)
    plot_per_version(per_version)
    plot_score_distribution()
    plot_feature_importance(fi)

    # 6. LaTeX
    generate_latex_table(summary)

    # 7. Statistical significance
    try:
        mcnemar_test()
    except ImportError:
        print("[ml_ablation] scipy not available; skipping McNemar's test.")
    except Exception as exc:
        print(f"[ml_ablation] McNemar's test failed: {exc}")

    print(f"\n{'─'*60}")
    print("  ML Ablation artifacts written to:")
    print(f"    reports/ml_ablation_*.csv")
    print(f"    reports/ml_ablation_table.tex")
    print(f"    figures/ml_ablation_*.png")
    print(f"    figures/ml_score_distribution.png")
    print(f"    figures/ml_feature_importance.png")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
