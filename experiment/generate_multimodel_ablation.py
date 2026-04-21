#!/usr/bin/env python3
"""
Multi-model ablation: compare HSR across ranker modes × ML algorithms.

══════════════════════════════════════════════════════════════════════════════
Paper 2: Self-Healing Test Automation via ML-Enhanced Locator Recovery
         Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Reads experiment results from multiple modes (heuristic, ml, hybrid) and
multiple ML backends (gradient_boosting, xgboost), and generates:
  - reports/multimodel_ablation_table.tex  (full comparison LaTeX table)
  - figures/multimodel_ablation.png        (grouped bar chart)
  - figures/multimodel_per_version.png     (per-version breakdown)
  - reports/multimodel_summary.json        (machine-readable summary)

Expects results CSV files at:
  - reports/results_heuristic.csv
  - reports/results_ml.csv          (trained with GB)
  - reports/results_hybrid.csv      (trained with GB)
  - reports/results_ml_xgb.csv      (trained with XGBoost)
  - reports/results_hybrid_xgb.csv  (trained with XGBoost)

Usage:
  python experiment/generate_multimodel_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_results(path: Path) -> pd.DataFrame | None:
    """Load experiment results CSV if it exists."""
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_hsr(df: pd.DataFrame) -> dict:
    """Compute HSR, broken, healed from results DataFrame."""
    if df is None or len(df) == 0:
        return {"broken": 0, "healed": 0, "failed": 0, "hsr": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    # Determine status column name
    status_col = "status" if "status" in df.columns else "result"
    if status_col not in df.columns:
        # Try step_logs approach
        broken = len(df[df.get("had_broken_locator", df.get("broken", pd.Series(dtype=bool))) == True])
        healed = len(df[df.get("was_healed", df.get("healed", pd.Series(dtype=bool))) == True])
    else:
        # Count by status
        broken = 0
        healed = 0
        for _, row in df.iterrows():
            status = str(row.get(status_col, ""))
            if "heal_success" in status or "healed" in status:
                broken += 1
                healed += 1
            elif "heal_failed" in status or "failed" in status.lower():
                broken += 1

    if broken == 0:
        # Fallback: use step_logs if available
        try:
            import ast
            for _, row in df.iterrows():
                logs = row.get("step_logs", "[]")
                if isinstance(logs, str):
                    steps = ast.literal_eval(logs) if logs.startswith("[") else []
                else:
                    steps = []
                for step in steps:
                    if isinstance(step, dict):
                        s = step.get("status", "")
                        if s in ("heal_success", "heal_failed"):
                            broken += 1
                            if s == "heal_success":
                                healed += 1
        except Exception:
            pass

    failed = broken - healed
    hsr = healed / broken if broken > 0 else 0.0
    ci_low, ci_high = wilson_ci(healed, broken)

    return {
        "broken": broken,
        "healed": healed,
        "failed": failed,
        "hsr": round(hsr * 100, 1),
        "ci_low": round(ci_low * 100, 1),
        "ci_high": round(ci_high * 100, 1),
    }


def compute_per_version_hsr(df: pd.DataFrame) -> dict[str, dict]:
    """Compute HSR per UI version."""
    if df is None:
        return {}

    version_col = "ui_version" if "ui_version" in df.columns else "version"
    if version_col not in df.columns:
        return {}

    per_version = {}
    for version in sorted(df[version_col].unique()):
        if "version_1" in str(version):
            continue  # Skip baseline
        vdf = df[df[version_col] == version]
        per_version[str(version)] = compute_hsr(vdf)

    return per_version


def main():
    print("=" * 70)
    print("  Multi-Model Ablation Analysis")
    print("  Paper 2: Self-Healing via ML-Enhanced Locator Recovery")
    print("=" * 70)

    # ── Define expected result files ──────────────────────────────────────
    modes = [
        {"name": "Heuristic", "file": "results_heuristic.csv", "algo": "—"},
        {"name": "ML (GB)", "file": "results_ml.csv", "algo": "Gradient Boosting"},
        {"name": "Hybrid (GB)", "file": "results_hybrid.csv", "algo": "Gradient Boosting"},
        {"name": "ML (XGB)", "file": "results_ml_xgb.csv", "algo": "XGBoost"},
        {"name": "Hybrid (XGB)", "file": "results_hybrid_xgb.csv", "algo": "XGBoost"},
    ]

    # Also check if default results.csv exists for heuristic
    heuristic_path = REPORTS / "results_heuristic.csv"
    if not heuristic_path.exists() and (REPORTS / "results.csv").exists():
        modes[0]["file"] = "results.csv"

    # ── Load and compute ─────────────────────────────────────────────────
    all_results = []
    for mode in modes:
        path = REPORTS / mode["file"]
        df = load_results(path)
        if df is not None:
            stats = compute_hsr(df)
            per_version = compute_per_version_hsr(df)
            entry = {**mode, **stats, "per_version": per_version, "available": True}
            print(f"  ✓ {mode['name']:20s} : HSR = {stats['hsr']:.1f}% [{stats['ci_low']:.1f}, {stats['ci_high']:.1f}] (n={stats['broken']})")
        else:
            entry = {**mode, "broken": 0, "healed": 0, "failed": 0,
                    "hsr": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                    "per_version": {}, "available": False}
            print(f"  ✗ {mode['name']:20s} : not found ({path.name})")
        all_results.append(entry)

    available = [r for r in all_results if r["available"]]
    if len(available) < 2:
        print("\n[WARN] Need at least 2 result sets for comparison. Run experiments first.")
        print("Commands to generate results:")
        print("  python run_experiment.py --healer-mode heuristic --results-file reports/results_heuristic.csv")
        print("  # Copy models/healing_ranker_gb.pkl → models/healing_ranker.pkl, then:")
        print("  python run_experiment.py --healer-mode ml --results-file reports/results_ml.csv")
        print("  python run_experiment.py --healer-mode hybrid --results-file reports/results_hybrid.csv")
        print("  # Copy models/healing_ranker_xgb.pkl → models/healing_ranker.pkl, then:")
        print("  python run_experiment.py --healer-mode ml --results-file reports/results_ml_xgb.csv")
        print("  python run_experiment.py --healer-mode hybrid --results-file reports/results_hybrid_xgb.csv")

    # ── Generate LaTeX table ──────────────────────────────────────────────
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Multi-Model Ablation: HSR by Ranker Mode and ML Algorithm (Wilson 95\% CI)}",
        r"\label{tab:multimodel_ablation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Mode & Algorithm & Broken & Healed & Failed & HSR & 95\% CI \\",
        r"\midrule",
    ]
    for r in all_results:
        if r["available"]:
            tex.append(
                f"{r['name']} & {r['algo']} & {r['broken']:,} & {r['healed']:,} & "
                f"{r['failed']:,} & {r['hsr']:.1f}\\% & [{r['ci_low']:.1f}, {r['ci_high']:.1f}] \\\\"
            )
        else:
            tex.append(f"{r['name']} & {r['algo']} & --- & --- & --- & --- & --- \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = REPORTS / "multimodel_ablation_table.tex"
    tex_path.write_text("\n".join(tex))
    print(f"\nLaTeX table saved to {tex_path}")

    # ── Generate summary JSON ─────────────────────────────────────────────
    summary = {r["name"]: {k: v for k, v in r.items() if k != "per_version"} for r in all_results}
    json_path = REPORTS / "multimodel_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary saved to {json_path}")

    # ── Generate figures ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        avail = [r for r in all_results if r["available"]]
        if len(avail) < 2:
            print("[WARN] Not enough data for figures.")
            return

        # ── Figure 1: HSR comparison bar chart ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["name"] for r in avail]
        hsrs = [r["hsr"] for r in avail]
        ci_lows = [r["hsr"] - r["ci_low"] for r in avail]
        ci_highs = [r["ci_high"] - r["hsr"] for r in avail]

        colors = []
        for r in avail:
            if "Heuristic" in r["name"]:
                colors.append("#607D8B")
            elif "XGB" in r["name"]:
                colors.append("#FF9800")
            elif "GB" in r["name"]:
                colors.append("#2196F3")
            else:
                colors.append("#9E9E9E")

        bars = ax.bar(names, hsrs, color=colors, alpha=0.85,
                     yerr=[ci_lows, ci_highs], capsize=5, ecolor="black")

        for bar, hsr_val in zip(bars, hsrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f"{hsr_val:.1f}%", ha="center", va="bottom", fontweight="bold")

        ax.set_ylabel("Healing Success Rate (%)")
        ax.set_title("Multi-Model Ablation: HSR by Ranker Mode and ML Algorithm\nWilson 95% CI Error Bars",
                     fontweight="bold")
        ax.set_ylim(0, max(hsrs) + 10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        fig_path = FIGURES / "multimodel_ablation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Ablation figure saved to {fig_path}")

        # ── Figure 2: Per-version comparison ──────────────────────────────
        modes_with_versions = [r for r in avail if r.get("per_version")]
        if modes_with_versions:
            versions = sorted(set(
                v for r in modes_with_versions for v in r["per_version"]
            ))
            if versions:
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(versions))
                width = 0.8 / len(modes_with_versions)

                for i, r in enumerate(modes_with_versions):
                    hsr_vals = [r["per_version"].get(v, {}).get("hsr", 0) for v in versions]
                    offset = (i - len(modes_with_versions)/2 + 0.5) * width
                    color = colors[avail.index(r)] if avail.index(r) < len(colors) else "#9E9E9E"
                    ax.bar(x + offset, hsr_vals, width, label=r["name"], color=color, alpha=0.85)

                ax.set_xticks(x)
                ax.set_xticklabels([v.replace("version_", "v") for v in versions])
                ax.set_ylabel("HSR (%)")
                ax.set_title("Per-Version HSR Across Ranker Modes and ML Algorithms", fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()

                fig_path = FIGURES / "multimodel_per_version.png"
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Per-version figure saved to {fig_path}")

    except ImportError:
        print("[WARN] matplotlib not available; skipping figures.")

    print("\nMulti-model ablation analysis complete.")


if __name__ == "__main__":
    main()
