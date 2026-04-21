#!/usr/bin/env python3
"""
Train both Gradient Boosting and XGBoost models, save separately, and compare.

══════════════════════════════════════════════════════════════════════════════
Paper 2: Self-Healing Test Automation via ML-Enhanced Locator Recovery
         Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Trains both algorithms with identical data/splits and produces:
  - models/healing_ranker_gb.pkl           (Gradient Boosting model)
  - models/healing_ranker_gb.metrics.json
  - models/healing_ranker_xgb.pkl          (XGBoost model)
  - models/healing_ranker_xgb.metrics.json
  - models/model_comparison.json           (side-by-side metrics)
  - reports/model_comparison_table.tex     (LaTeX-ready table)
  - figures/model_comparison.png           (visual comparison)

After running this script, use the best model by copying it to
models/healing_ranker.pkl, or pass --use-best to auto-select.

Usage:
  python scripts/train_multimodel.py
  python scripts/train_multimodel.py --use-best   # auto-copy best → healing_ranker.pkl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_training_data import FEATURE_COLUMNS


def train_both(data_path: str, use_best: bool = False) -> dict:
    """Train both GB and XGBoost, compare, and optionally select best."""
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix,
    )

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        use_smote = True
    except ImportError:
        use_smote = False
        ImbPipeline = Pipeline

    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        print("[ERROR] XGBoost is not installed. Cannot run multi-model comparison.")
        print("Install: pip install xgboost")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df["label"].values.astype(int)
    print(f"Dataset: {len(df):,} instances | Pos: {y.sum():,} | Neg: {(1-y).sum():,}\n")

    # Same split for both
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    models_info = {
        "gradient_boosting": {
            "cls": GradientBoostingClassifier,
            "kwargs": dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                          random_state=42, min_samples_split=20, min_samples_leaf=10,
                          subsample=0.8),
            "file": "healing_ranker_gb.pkl",
            "label": "Gradient Boosting",
        },
        "xgboost": {
            "cls": XGBClassifier,
            "kwargs": dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                          random_state=42, use_label_encoder=False,
                          eval_metric="logloss", subsample=0.8,
                          colsample_bytree=0.8, verbosity=0),
            "file": "healing_ranker_xgb.pkl",
            "label": "XGBoost",
        },
    }

    for algo_name, info in models_info.items():
        print(f"{'═'*60}")
        print(f"  Training: {info['label']}")
        print(f"{'═'*60}")
        t0 = time.time()

        model = info["cls"](**info["kwargs"])
        steps = []
        if use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        pipeline = ImbPipeline(steps)

        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Evaluate
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # Cross-validation
        try:
            cv_auc = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
            cv_mean, cv_std = cv_auc.mean(), cv_auc.std()
        except Exception:
            cv_mean, cv_std = 0.0, 0.0

        # Feature importance
        gb_model = pipeline.named_steps["model"]
        importances = gb_model.feature_importances_
        feat_imp = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, importances)}

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(f"  AUC     : {auc:.6f}")
        print(f"  F1      : {f1:.6f}")
        print(f"  Prec    : {prec:.6f}")
        print(f"  Recall  : {rec:.6f}")
        print(f"  CV AUC  : {cv_mean:.6f} ± {cv_std:.6f}")
        print(f"  Time    : {elapsed:.1f}s")
        print(f"  CM      : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        print()

        # Top 5 features
        print(f"  Top 5 features:")
        sorted_feats = sorted(feat_imp.items(), key=lambda x: -x[1])
        for feat, imp in sorted_feats[:5]:
            bar = "█" * int(imp * 40)
            print(f"    {feat:28s} {imp:.4f}  {bar}")
        print()

        # Save model
        output = PROJECT_ROOT / "models" / info["file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        pipeline._governance_meta = {
            "model_version": f"healing-{algo_name}-v1",
            "algorithm": algo_name,
            "auc": round(auc, 6),
            "f1_macro": round(f1, 6),
            "features": FEATURE_COLUMNS,
            "author": "Vijay P. Javvadi",
        }
        joblib.dump(pipeline, output)
        print(f"  Model saved: {output}")

        metrics = {
            "algorithm": algo_name,
            "label": info["label"],
            "auc": round(auc, 6),
            "f1_macro": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "cv_auc_mean": round(cv_mean, 6),
            "cv_auc_std": round(cv_std, 6),
            "training_time_sec": round(elapsed, 2),
            "training_instances": len(X_train),
            "test_instances": len(X_test),
            "confusion_matrix": {"TN": int(cm[0][0]), "FP": int(cm[0][1]),
                                "FN": int(cm[1][0]), "TP": int(cm[1][1])},
            "feature_importance": feat_imp,
            "model_path": str(output),
        }

        metrics_path = output.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"  Metrics saved: {metrics_path}\n")

        results[algo_name] = metrics

    # ── Comparison ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'═'*60}")
    print(f"{'Metric':<20} {'Gradient Boosting':>20} {'XGBoost':>20} {'Delta':>12}")
    print(f"{'─'*72}")

    gb = results["gradient_boosting"]
    xgb = results["xgboost"]

    for metric in ["auc", "f1_macro", "precision", "recall", "cv_auc_mean", "training_time_sec"]:
        g = gb[metric]
        x = xgb[metric]
        d = x - g
        sign = "+" if d > 0 else ""
        fmt = ".6f" if metric != "training_time_sec" else ".1f"
        print(f"  {metric:<20} {g:>20{fmt}} {x:>20{fmt}} {sign}{d:>11{fmt}}")

    # ── Save comparison JSON ──────────────────────────────────────────────
    comparison = {
        "gradient_boosting": gb,
        "xgboost": xgb,
        "winner": "xgboost" if xgb["auc"] > gb["auc"] else "gradient_boosting",
        "auc_delta": round(xgb["auc"] - gb["auc"], 6),
        "f1_delta": round(xgb["f1_macro"] - gb["f1_macro"], 6),
    }
    comp_path = PROJECT_ROOT / "models" / "model_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to {comp_path}")

    # ── Generate LaTeX table ──────────────────────────────────────────────
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Model Comparison: Gradient Boosting vs.\ XGBoost on Healing Candidate Classification}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Gradient Boosting & XGBoost \\",
        r"\midrule",
        f"AUC (test) & {gb['auc']:.4f} & {xgb['auc']:.4f} \\\\",
        f"F1 (macro) & {gb['f1_macro']:.4f} & {xgb['f1_macro']:.4f} \\\\",
        f"Precision & {gb['precision']:.4f} & {xgb['precision']:.4f} \\\\",
        f"Recall & {gb['recall']:.4f} & {xgb['recall']:.4f} \\\\",
        f"CV AUC (5-fold) & {gb['cv_auc_mean']:.4f} $\\pm$ {gb['cv_auc_std']:.4f} & {xgb['cv_auc_mean']:.4f} $\\pm$ {xgb['cv_auc_std']:.4f} \\\\",
        f"Training time (s) & {gb['training_time_sec']:.1f} & {xgb['training_time_sec']:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = PROJECT_ROOT / "reports" / "model_comparison_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex))
    print(f"LaTeX table saved to {tex_path}")

    # ── Generate comparison figure ────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Gradient Boosting vs. XGBoost: Model Comparison", fontsize=14, fontweight="bold")

        # Panel 1: Metric comparison bar chart
        metrics_to_plot = ["auc", "f1_macro", "precision", "recall"]
        gb_vals = [gb[m] for m in metrics_to_plot]
        xgb_vals = [xgb[m] for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35
        axes[0].bar(x - width/2, gb_vals, width, label="Gradient Boosting", color="#2196F3", alpha=0.8)
        axes[0].bar(x + width/2, xgb_vals, width, label="XGBoost", color="#FF9800", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(["AUC", "F1", "Precision", "Recall"])
        axes[0].set_ylim(0.90, 1.01)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Classification Metrics")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Panel 2: Feature importance comparison
        gb_imp = gb["feature_importance"]
        xgb_imp = xgb["feature_importance"]
        # Top 8 features by average importance
        avg_imp = {f: (gb_imp.get(f, 0) + xgb_imp.get(f, 0)) / 2 for f in FEATURE_COLUMNS}
        top_feats = sorted(avg_imp, key=lambda x: avg_imp[x], reverse=True)[:8]

        y_pos = np.arange(len(top_feats))
        gb_top = [gb_imp.get(f, 0) for f in top_feats]
        xgb_top = [xgb_imp.get(f, 0) for f in top_feats]

        axes[1].barh(y_pos + 0.2, gb_top, 0.35, label="GB", color="#2196F3", alpha=0.8)
        axes[1].barh(y_pos - 0.2, xgb_top, 0.35, label="XGB", color="#FF9800", alpha=0.8)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([f.replace("_", " ") for f in top_feats], fontsize=8)
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Feature Importance (Top 8)")
        axes[1].legend(fontsize=8)
        axes[1].invert_yaxis()

        # Panel 3: Training time comparison
        times = [gb["training_time_sec"], xgb["training_time_sec"]]
        colors = ["#2196F3", "#FF9800"]
        bars = axes[2].bar(["Gradient\nBoosting", "XGBoost"], times, color=colors, alpha=0.8)
        for bar, t in zip(bars, times):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{t:.1f}s", ha="center", va="bottom", fontweight="bold")
        axes[2].set_ylabel("Training Time (seconds)")
        axes[2].set_title("Training Speed")
        axes[2].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = PROJECT_ROOT / "figures" / "model_comparison.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison figure saved to {fig_path}")
    except ImportError:
        print("[WARN] matplotlib not available; skipping figure generation.")

    # ── Auto-select best ──────────────────────────────────────────────────
    if use_best:
        winner = comparison["winner"]
        src = PROJECT_ROOT / "models" / models_info[winner]["file"]
        dst = PROJECT_ROOT / "models" / "healing_ranker.pkl"
        shutil.copy2(src, dst)
        # Also copy metrics
        src_m = src.with_suffix(".metrics.json")
        dst_m = dst.with_suffix(".metrics.json")
        shutil.copy2(src_m, dst_m)
        print(f"\n★ Best model ({winner}) copied to {dst}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train and compare GB vs XGBoost.")
    parser.add_argument(
        "--data", default=str(PROJECT_ROOT / "data" / "training" / "healing_training_data.csv"),
    )
    parser.add_argument("--use-best", action="store_true",
                       help="Auto-copy best model → models/healing_ranker.pkl")
    args = parser.parse_args()

    import numpy as np  # verify available
    train_both(args.data, use_best=args.use_best)
    print("\nMulti-model training complete.")


if __name__ == "__main__":
    main()
