#!/usr/bin/env python3
"""
Hyperparameter sweep for healing ranker — compare Gradient Boosting vs XGBoost.

══════════════════════════════════════════════════════════════════════════════
Paper 2: Self-Healing Test Automation via ML-Enhanced Locator Recovery
         Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Runs a grid search over key hyperparameters for both algorithms, evaluates
each combination via 5-fold stratified CV, and produces:
  - models/sweep_results.csv       (per-combo metrics)
  - models/best_model_summary.json (best config per algorithm)
  - figures/hp_sweep_heatmap.png   (visual comparison)
  - reports/hp_sweep_table.tex     (LaTeX-ready table)

Usage:
  python scripts/hyperparameter_sweep.py
  python scripts/hyperparameter_sweep.py --quick   # reduced grid for fast iteration
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product as iterproduct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_training_data import FEATURE_COLUMNS


def check_deps():
    for pkg in ["pandas", "sklearn", "numpy", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Missing: {pkg}. Install with pip.")
            sys.exit(1)


def run_sweep(data_path: str, quick: bool = False) -> list[dict]:
    """Run hyperparameter sweep across algorithms and param combos."""
    check_deps()

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score

    # Try SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        use_smote = True
    except ImportError:
        use_smote = False
        ImbPipeline = Pipeline

    # Try XGBoost
    xgb_available = False
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        print("[WARN] XGBoost not installed. Sweeping only Gradient Boosting.")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df["label"].values.astype(int)
    print(f"Dataset: {len(df):,} instances | Positive: {y.sum():,} | Negative: {(1-y).sum():,}")

    # ── Define search grid ─────────────────────────────────────────────────
    if quick:
        n_estimators_grid = [100, 200]
        max_depth_grid = [3, 5]
        learning_rate_grid = [0.05, 0.1]
    else:
        n_estimators_grid = [100, 200, 300, 500]
        max_depth_grid = [3, 4, 5, 6, 7]
        learning_rate_grid = [0.01, 0.05, 0.1, 0.15, 0.2]

    algorithms = ["gradient_boosting"]
    if xgb_available:
        algorithms.append("xgboost")

    total_combos = len(algorithms) * len(n_estimators_grid) * len(max_depth_grid) * len(learning_rate_grid)
    print(f"\nTotal combinations: {total_combos}")
    print(f"Algorithms: {algorithms}")
    print(f"n_estimators: {n_estimators_grid}")
    print(f"max_depth: {max_depth_grid}")
    print(f"learning_rate: {learning_rate_grid}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        "auc": "roc_auc",
        "f1": make_scorer(f1_score, average="macro"),
    }

    results = []
    idx = 0

    for algo, n_est, depth, lr in iterproduct(algorithms, n_estimators_grid, max_depth_grid, learning_rate_grid):
        idx += 1
        t0 = time.time()

        # Build model
        if algo == "xgboost":
            model = XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                random_state=42, use_label_encoder=False, eval_metric="logloss",
                subsample=0.8, colsample_bytree=0.8, verbosity=0,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                random_state=42, min_samples_split=20, min_samples_leaf=10,
                subsample=0.8,
            )

        # Build pipeline
        steps = []
        if use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        pipeline = ImbPipeline(steps)

        # Cross-validate
        try:
            cv_results = cross_validate(
                pipeline, X, y, cv=cv, scoring=scorers, n_jobs=-1,
                return_train_score=False,
            )
            auc_mean = cv_results["test_auc"].mean()
            auc_std = cv_results["test_auc"].std()
            f1_mean = cv_results["test_f1"].mean()
            f1_std = cv_results["test_f1"].std()
            status = "ok"
        except Exception as e:
            auc_mean = auc_std = f1_mean = f1_std = 0.0
            status = f"error: {e}"

        elapsed = time.time() - t0

        row = {
            "algorithm": algo,
            "n_estimators": n_est,
            "max_depth": depth,
            "learning_rate": lr,
            "cv_auc_mean": round(auc_mean, 6),
            "cv_auc_std": round(auc_std, 6),
            "cv_f1_mean": round(f1_mean, 6),
            "cv_f1_std": round(f1_std, 6),
            "elapsed_sec": round(elapsed, 2),
            "status": status,
        }
        results.append(row)

        marker = "★" if auc_mean > 0.999 else " "
        print(f"  [{idx:3d}/{total_combos}] {marker} {algo:20s} n={n_est:3d} d={depth} lr={lr:.2f} → AUC={auc_mean:.4f}±{auc_std:.4f}  F1={f1_mean:.4f}±{f1_std:.4f}  ({elapsed:.1f}s)")

    return results


def generate_artifacts(results: list[dict]) -> None:
    """Generate CSV, JSON, figure, and LaTeX table from sweep results."""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(results)

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = PROJECT_ROOT / "models" / "sweep_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved to {csv_path}")

    # ── Best per algorithm ─────────────────────────────────────────────────
    summary = {}
    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        best_idx = algo_df["cv_auc_mean"].idxmax()
        best = algo_df.loc[best_idx]
        summary[algo] = {
            "n_estimators": int(best["n_estimators"]),
            "max_depth": int(best["max_depth"]),
            "learning_rate": float(best["learning_rate"]),
            "cv_auc_mean": float(best["cv_auc_mean"]),
            "cv_auc_std": float(best["cv_auc_std"]),
            "cv_f1_mean": float(best["cv_f1_mean"]),
            "cv_f1_std": float(best["cv_f1_std"]),
        }
        print(f"\n  Best {algo}: n={best['n_estimators']} d={best['max_depth']} lr={best['learning_rate']} → AUC={best['cv_auc_mean']:.4f} F1={best['cv_f1_mean']:.4f}")

    json_path = PROJECT_ROOT / "models" / "best_model_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Best model summary saved to {json_path}")

    # ── LaTeX table: top 5 per algorithm ──────────────────────────────────
    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Hyperparameter Sweep: Top 5 Configurations per Algorithm (5-Fold Stratified CV)}",
        r"\label{tab:hp_sweep}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Algorithm & $n$ & Depth & LR & AUC & $\pm$ & F1 & $\pm$ \\",
        r"\midrule",
    ]

    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo].nlargest(5, "cv_auc_mean")
        algo_label = "Gradient Boosting" if algo == "gradient_boosting" else "XGBoost"
        for i, (_, row) in enumerate(algo_df.iterrows()):
            prefix = algo_label if i == 0 else ""
            tex_lines.append(
                f"{prefix} & {int(row['n_estimators'])} & {int(row['max_depth'])} & "
                f"{row['learning_rate']:.2f} & {row['cv_auc_mean']:.4f} & "
                f"{row['cv_auc_std']:.4f} & {row['cv_f1_mean']:.4f} & {row['cv_f1_std']:.4f} \\\\"
            )
        if algo != df["algorithm"].unique()[-1]:
            tex_lines.append(r"\midrule")

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = PROJECT_ROOT / "reports" / "hp_sweep_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines))
    print(f"LaTeX table saved to {tex_path}")

    # ── Heatmap figure ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(df["algorithm"].unique()), figsize=(7 * len(df["algorithm"].unique()), 5), squeeze=False)
        fig.suptitle("Hyperparameter Sweep: 5-Fold CV AUC by Depth × Learning Rate", fontsize=14, fontweight="bold")

        for ax_idx, algo in enumerate(sorted(df["algorithm"].unique())):
            ax = axes[0, ax_idx]
            algo_df = df[df["algorithm"] == algo]

            # Aggregate across n_estimators (take best for each depth × lr)
            pivot = algo_df.groupby(["max_depth", "learning_rate"])["cv_auc_mean"].max().reset_index()
            pivot_table = pivot.pivot(index="max_depth", columns="learning_rate", values="cv_auc_mean")

            im = ax.imshow(pivot_table.values, cmap="YlOrRd", aspect="auto",
                          vmin=df["cv_auc_mean"].min(), vmax=df["cv_auc_mean"].max())

            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels([f"{lr:.2f}" for lr in pivot_table.columns])
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Max Depth")
            algo_title = "Gradient Boosting" if algo == "gradient_boosting" else "XGBoost"
            ax.set_title(algo_title)

            # Annotate cells
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    val = pivot_table.values[i, j]
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                           color="white" if val > 0.997 else "black", fontsize=8)

        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="CV AUC")
        plt.tight_layout()
        fig_path = PROJECT_ROOT / "figures" / "hp_sweep_heatmap.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to {fig_path}")
    except Exception as e:
        print(f"[WARN] Could not generate heatmap: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for healing ranker.")
    parser.add_argument(
        "--data", default=str(PROJECT_ROOT / "data" / "training" / "healing_training_data.csv"),
        help="Path to training CSV.",
    )
    parser.add_argument("--quick", action="store_true", help="Use reduced grid for fast iteration.")
    args = parser.parse_args()

    results = run_sweep(args.data, quick=args.quick)
    generate_artifacts(results)
    print("\nHyperparameter sweep complete.")


if __name__ == "__main__":
    main()
