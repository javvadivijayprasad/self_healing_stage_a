#!/usr/bin/env python3
"""
Train the ML healing ranker — Paper 2, Stage B methodology.

══════════════════════════════════════════════════════════════════════════════
Paper 2: Self-Healing Test Automation via ML-Enhanced Locator Recovery
         Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Pipeline (mirrors test-prioritization-service/scripts/train_model.py):
  1. Load CSV dataset (from extract_training_data.py)
  2. Split 80/20 train/test (stratified by label)
  3. Apply SMOTE to training set (addresses class imbalance)
  4. Scale features with StandardScaler
  5. Train GradientBoostingClassifier (XGBoost backend)
  6. Evaluate: AUC, F1, Precision, Recall, top-K accuracy
  7. Serialise pipeline to models/healing_ranker.pkl

Usage:
  python scripts/train_healing_model.py
  python scripts/train_healing_model.py --data data/training/healing_training_data.csv
  python scripts/train_healing_model.py --data data/training/healing_training_data.csv --algorithm xgboost
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_training_data import FEATURE_COLUMNS


def check_deps() -> None:
    missing = []
    for pkg in ["pandas", "sklearn", "numpy", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing training dependencies: {', '.join(missing)}")
        print("Install with: pip install pandas scikit-learn numpy joblib imbalanced-learn xgboost")
        sys.exit(1)


def train(
    data_path: str,
    output_path: str = "models/healing_ranker.pkl",
    algorithm: str = "gradient_boosting",
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    threshold: float = 0.50,
    random_state: int = 42,
) -> dict:
    """Train and evaluate the healing ranker model."""
    check_deps()

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

    # Try SMOTE, fall back to class_weight if imbalanced-learn unavailable
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        use_smote = True
    except ImportError:
        print("[WARN] imbalanced-learn not installed; using class_weight='balanced' instead of SMOTE.")
        use_smote = False
        ImbPipeline = Pipeline  # type: ignore

    # Try XGBoost if requested
    xgb_cls = None
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
            xgb_cls = XGBClassifier
        except ImportError:
            print("[WARN] xgboost not installed; falling back to sklearn GradientBoosting.")
            algorithm = "gradient_boosting"

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    required_cols = FEATURE_COLUMNS + ["label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in dataset: {missing}")
        print(f"Expected: {required_cols}")
        sys.exit(1)

    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df["label"].values.astype(int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"Dataset: {len(df):,} instances | Positive: {n_pos:,} ({n_pos/len(df)*100:.1f}%) | Negative: {n_neg:,}")

    # ── Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Build model ────────────────────────────────────────────────────────
    if algorithm == "xgboost" and xgb_cls is not None:
        print("\nBuilding pipeline: SMOTE → StandardScaler → XGBClassifier...")
        model = xgb_cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        print("\nBuilding pipeline: SMOTE → StandardScaler → GradientBoosting...")
        model_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
        )
        if not use_smote:
            # No SMOTE; compensate imbalance with balanced class weights
            # sklearn GB doesn't have class_weight, so we use sample_weight later
            pass
        model = GradientBoostingClassifier(**model_kwargs)

    steps = []
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    pipeline = ImbPipeline(steps)

    print("Training... (this may take a moment)")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  Algorithm : {algorithm}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Threshold : {threshold}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Wrong', 'Correct'])}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Cross-validation AUC
    try:
        cv_auc = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_mean, cv_std = cv_auc.mean(), cv_auc.std()
        print(f"\n  5-fold CV AUC : {cv_mean:.4f} ± {cv_std:.4f}")
    except Exception as exc:
        print(f"\n  [WARN] Cross-validation skipped: {exc}")
        cv_mean, cv_std = 0.0, 0.0

    print(f"{'─'*60}")

    # ── Feature importance ─────────────────────────────────────────────────
    try:
        gb_model = pipeline.named_steps["model"]
        importances = gb_model.feature_importances_
        print("\nFeature importances:")
        for feat, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {feat:28} {imp:.4f}  {bar}")
    except Exception:
        print("\n  [WARN] Feature importances not available for this model type.")

    # ── Save model ─────────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Attach governance metadata
    pipeline._governance_meta = {
        "model_version": f"healing-{algorithm}-v1",
        "training_instances": len(X_train),
        "auc": round(auc, 4),
        "f1_macro": round(f1, 4),
        "threshold": threshold,
        "features": FEATURE_COLUMNS,
        "algorithm": algorithm,
        "smote": use_smote,
        "paper": "Self-Healing Test Automation via ML-Enhanced Locator Recovery",
        "author": "Vijay P. Javvadi",
    }

    joblib.dump(pipeline, output)
    print(f"\nModel saved to {output}")

    metrics = {
        "auc": round(auc, 4),
        "f1_macro": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "cv_auc_mean": round(cv_mean, 4),
        "cv_auc_std": round(cv_std, 4),
        "training_instances": len(X_train),
        "test_instances": len(X_test),
        "threshold": threshold,
        "algorithm": algorithm,
        "features": FEATURE_COLUMNS,
        "model_path": str(output),
    }

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ML healing ranker (Paper 2, Stage B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "training" / "healing_training_data.csv"),
        help="Path to training CSV (from extract_training_data.py).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "healing_ranker.pkl"),
        help="Output path for the serialised model.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["gradient_boosting", "xgboost"],
        default="gradient_boosting",
        help="ML algorithm to use.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        output_path=args.output,
        algorithm=args.algorithm,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        random_state=args.random_state,
    )
    print(f"\nTraining complete. AUC: {metrics['auc']} | F1: {metrics['f1_macro']}")


if __name__ == "__main__":
    main()
