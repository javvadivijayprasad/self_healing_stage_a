from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
import time
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# Run a pipeline step
# --------------------------------------------------

def run_step(title: str, command: list[str]) -> None:

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = subprocess.run(command, cwd=BASE_DIR)

    if result.returncode != 0:
        raise SystemExit(f"{title} failed with exit code {result.returncode}")


# --------------------------------------------------
# Archive previous outputs
# --------------------------------------------------

def archive_previous_outputs() -> None:

    reports_dir = BASE_DIR / "reports"
    figures_dir = BASE_DIR / "figures"
    history_dir = reports_dir / "history"

    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = history_dir / f"run_{timestamp}"

    moved = False

    for file in reports_dir.glob("*.csv"):
        archive_dir.mkdir(exist_ok=True)
        shutil.move(str(file), archive_dir / file.name)
        moved = True

    for file in figures_dir.glob("*.png"):
        archive_dir.mkdir(exist_ok=True)
        shutil.move(str(file), archive_dir / file.name)
        moved = True

    if moved:
        print(f"[INFO] Archived previous results → {archive_dir}")
    else:
        print("[INFO] No previous outputs to archive.")


# --------------------------------------------------
# Validate dataset integrity
# --------------------------------------------------

def validate_results() -> None:

    results_file = BASE_DIR / "reports" / "results.csv"

    if not results_file.exists():
        raise SystemExit("results.csv not generated — experiment failed")

    with open(results_file, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        if not rows:
            raise SystemExit("results.csv is empty — experiment invalid")

        required_columns = [
            "run_id",
            "ui_version",
            "test_name",
            "healed_steps",
            "baseline_steps",
            "failed_steps"
        ]

        for col in required_columns:
            if col not in reader.fieldnames:
                raise SystemExit(f"results.csv missing column: {col}")

    print(f"[INFO] Dataset validated ({len(rows)} experiment rows)")


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def validate_ml_training_data() -> None:
    """Validate that ML training data was extracted successfully."""

    training_file = BASE_DIR / "data" / "training" / "healing_training_data.csv"

    if not training_file.exists():
        raise SystemExit("healing_training_data.csv not generated — extraction failed")

    with open(training_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        if not rows:
            raise SystemExit("healing_training_data.csv is empty — extraction invalid")

        required_columns = [
            "tag_match", "text_similarity", "attribute_similarity",
            "class_similarity", "parent_similarity", "depth_similarity",
            "id_similarity", "name_similarity", "placeholder_similarity",
            "aria_label_similarity", "type_attr_match", "sibling_count_ratio",
            "multi_attr_match_count", "label",
        ]

        for col in required_columns:
            if col not in reader.fieldnames:
                raise SystemExit(f"healing_training_data.csv missing column: {col}")

    n_pos = sum(1 for r in rows if r["label"] == "1")
    n_neg = sum(1 for r in rows if r["label"] == "0")
    print(f"[INFO] Training data validated ({len(rows)} rows, {n_pos} positive, {n_neg} negative)")


def validate_trained_model() -> None:
    """Validate that the ML model was trained and saved successfully."""

    model_file = BASE_DIR / "models" / "healing_ranker.pkl"
    metrics_file = BASE_DIR / "models" / "healing_ranker.metrics.json"

    if not model_file.exists():
        raise SystemExit("healing_ranker.pkl not generated — training failed")

    if not metrics_file.exists():
        raise SystemExit("healing_ranker.metrics.json not generated — training failed")

    import json
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    auc = metrics.get("auc", 0)
    f1 = metrics.get("f1_macro", 0)
    print(f"[INFO] Trained model validated (AUC={auc}, F1={f1})")

    if auc < 0.50:
        print(f"[WARN] Model AUC ({auc}) below 0.50 — model may not outperform random.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full self-healing experiment pipeline."
    )
    parser.add_argument(
        "--with-ablation",
        action="store_true",
        help="Also run the rule_based, random, and none healer-mode "
             "experiments after the heuristic run so the mode-comparison "
             "table is populated. Adds ~3x to total wall time.",
    )
    parser.add_argument(
        "--with-ml",
        action="store_true",
        help="Train the ML ranker from heuristic experiment data, then "
             "run ml and hybrid ablation experiments and generate the "
             "ML ablation analysis. Adds ~2x to total wall time.",
    )
    parser.add_argument(
        "--ablation-runs",
        type=int,
        default=50,
        help="Experiment runs per ablation mode (default 50).",
    )
    parser.add_argument(
        "--ml-algorithm",
        choices=["gradient_boosting", "xgboost"],
        default="gradient_boosting",
        help="ML algorithm for the healing ranker (default: gradient_boosting).",
    )
    parser.add_argument(
        "--with-multimodel",
        action="store_true",
        help="Train both GB and XGBoost, run experiments with each, "
             "and generate a multi-model comparison ablation. "
             "Implies --with-ml. Adds ~4x to total wall time.",
    )
    parser.add_argument(
        "--with-hp-sweep",
        action="store_true",
        help="Run hyperparameter sweep across algorithms before training. "
             "Adds significant wall time but finds optimal configs.",
    )
    parser.add_argument(
        "--retrain-from-events",
        action="store_true",
        help="Accumulate all heal event data (experiment + real-time + production) "
             "and retrain the model on the combined dataset.",
    )
    return parser.parse_args()


def main() -> None:

    args = _parse_args()
    python_exe = sys.executable

    print("\n" + "=" * 60)
    print("SELF-HEALING LOCATOR EXPERIMENT PIPELINE")
    print("=" * 60)

    print("[INFO] Project root:", BASE_DIR)
    print("[INFO] Python:", python_exe)
    print("[INFO] Start time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[INFO] Ablation sweep:", "ENABLED" if args.with_ablation else "skipped")
    # --with-multimodel implies --with-ml
    if args.with_multimodel:
        args.with_ml = True

    print("[INFO] ML training:  ", "ENABLED" if args.with_ml else "skipped")
    print("[INFO] Multi-model:  ", "ENABLED" if args.with_multimodel else "skipped")
    print("[INFO] HP sweep:     ", "ENABLED" if args.with_hp_sweep else "skipped")
    if args.with_ml:
        print("[INFO] ML algorithm: ", args.ml_algorithm)

    archive_previous_outputs()

    # STEP 1 — Generate mutated UI versions
    run_step(
        "STEP 1 - Generate mutated UI versions",
        [python_exe, "dom_breaker/dom_mutation_generator.py"],
    )

    # STEP 2 — Analyze mutations
    run_step(
        "STEP 2 - Analyze DOM mutations",
        [python_exe, "healing/analyze_mutations.py"],
    )

    # STEP 3 — Run Selenium experiments (heuristic mode, canonical results.csv)
    run_step(
        "STEP 3 - Run self-healing experiments (heuristic)",
        [python_exe, "run_experiment.py", "--healer-mode", "heuristic"],
    )

    # STEP 3b — Ablation sweep (opt-in via --with-ablation)
    if args.with_ablation:
        for mode in ("rule_based", "random", "none"):
            run_step(
                f"STEP 3b - Ablation run ({mode})",
                [python_exe, "run_experiment.py",
                 "--healer-mode", mode,
                 "--runs", str(args.ablation_runs)],
            )

    # STEP 4 — Validate dataset
    validate_results()

    # ══════════════════════════════════════════════════════════════════
    # ML PIPELINE (opt-in via --with-ml)
    # Steps 4a-4f: Extract → Train → Validate → Experiment → Ablation
    # ══════════════════════════════════════════════════════════════════
    if args.with_ml:

        # STEP 4a — Extract training data from heuristic experiment results
        run_step(
            "STEP 4a - Extract ML training data from heuristic results",
            [python_exe, "scripts/extract_training_data.py"],
        )

        # STEP 4b — Validate extracted training data
        validate_ml_training_data()

        # STEP 4c — Train the ML healing ranker model
        run_step(
            "STEP 4c - Train ML healing ranker model",
            [python_exe, "scripts/train_healing_model.py",
             "--algorithm", args.ml_algorithm],
        )

        # STEP 4d — Validate trained model
        validate_trained_model()

        # STEP 4e — Run experiments with ML ranker
        run_step(
            "STEP 4e - Run self-healing experiments (ML ranker)",
            [python_exe, "run_experiment.py",
             "--healer-mode", "ml",
             "--runs", str(args.ablation_runs)],
        )

        # STEP 4f — Run experiments with Hybrid ranker
        run_step(
            "STEP 4f - Run self-healing experiments (Hybrid ranker)",
            [python_exe, "run_experiment.py",
             "--healer-mode", "hybrid",
             "--runs", str(args.ablation_runs)],
        )

    # STEP 5 — Analyze experiment results
    run_step(
        "STEP 5 - Analyze experiment results",
        [python_exe, "healing/analyze_results.py"],
    )

    # STEP 6 — Generate evaluation figures
    run_step(
        "STEP 6 - Generate evaluation figures",
        [python_exe, "experiment/generate_paper_figures.py"],
    )

    # STEP 7 — Architecture diagrams
    run_step(
        "STEP 7 - Generate architecture diagrams",
        [python_exe, "experiment/generate_architecture_figures.py"],
    )

    # STEP 8 — Locator healing report
    run_step(
        "STEP 8 - Generate locator healing report",
        [python_exe, "experiment/generate_locator_healing_report.py"],
    )

    # STEP 9 — Locator fragility analysis
    run_step(
        "STEP 9 - Locator fragility analysis",
        [python_exe, "experiment/generate_locator_fragility.py"],
    )

    # STEP 10 — Locator similarity heatmap
    run_step(
        "STEP 10 - Generate locator similarity heatmap",
        [python_exe, "experiment/generate_locator_similarity_heatmap.py"],
    )

    # STEP 11 — Mutation vs healing analysis
    run_step(
        "STEP 11 - Mutation vs healing analysis",
        [python_exe, "experiment/generate_mutation_vs_healing_table.py"],
    )

    # STEP 12 — Advanced analysis
    run_step(
        "STEP 12 - Advanced experiment analysis",
        [python_exe, "experiment/generate_advanced_analysis.py"],
    )

    run_step(
         "STEP 13 - Generate experiment summary",
         [python_exe, "experiment/generate_experiment_summary.py"],
    )

    # STEP 14 — Statistical analysis (Wilson CIs, chi-square, Kruskal-Wallis)
    run_step(
        "STEP 14 - Statistical analysis of HSR",
        [python_exe, "experiment/generate_statistical_analysis.py"],
    )

    # STEP 15 — Threshold sweep and denominator decomposition
    run_step(
        "STEP 15 - Threshold sweep and denominator decomposition",
        [python_exe, "experiment/generate_threshold_sweep.py"],
    )

    # STEP 16 — Ranker-mode ablation comparison (skipped silently if only
    # one mode has been run so far)
    run_step(
        "STEP 16 - Ranker mode ablation comparison",
        [python_exe, "experiment/generate_mode_comparison.py"],
    )

    # STEP 17 — Mutation density vs HSR (per-page)
    run_step(
        "STEP 17 - Mutation density analysis",
        [python_exe, "experiment/generate_mutation_density_analysis.py"],
    )

    # STEP 18 — Bootstrap CIs, permutation tests, Cliff's delta
    run_step(
        "STEP 18 - Bootstrap and effect-size analysis",
        [python_exe, "experiment/generate_bootstrap_effect_size.py"],
    )

    # STEP 19 — Maintenance cost model
    run_step(
        "STEP 19 - Maintenance cost model",
        [python_exe, "experiment/generate_cost_model.py"],
    )

    # STEP 20 — Mutation realism audit (static mapping to prior literature)
    run_step(
        "STEP 20 - Mutation realism audit",
        [python_exe, "experiment/generate_mutation_realism_audit.py"],
    )

    # ══════════════════════════════════════════════════════════════════
    # ML ABLATION ANALYSIS (runs only if --with-ml was set)
    # ══════════════════════════════════════════════════════════════════
    if args.with_ml:

        # STEP 21 — ML ablation: heuristic vs ML vs hybrid comparison
        run_step(
            "STEP 21 - ML ablation analysis (heuristic vs ML vs hybrid)",
            [python_exe, "experiment/generate_ml_ablation.py"],
        )

    # ══════════════════════════════════════════════════════════════════
    # MULTI-MODEL COMPARISON (opt-in via --with-multimodel)
    # Steps 22-25: HP sweep → Train both → XGBoost experiments → Ablation
    # ══════════════════════════════════════════════════════════════════
    if args.with_multimodel:

        # STEP 22 — Hyperparameter sweep (optional, via --with-hp-sweep)
        if args.with_hp_sweep:
            run_step(
                "STEP 22 - Hyperparameter sweep (GB vs XGBoost)",
                [python_exe, "scripts/hyperparameter_sweep.py"],
            )

        # STEP 23 — Train both models side by side
        run_step(
            "STEP 23 - Train GB and XGBoost models (multi-model comparison)",
            [python_exe, "scripts/train_multimodel.py", "--use-best"],
        )

        # STEP 24 — Run experiments with XGBoost model
        # First, copy XGBoost model into the active slot
        import shutil
        xgb_model = BASE_DIR / "models" / "healing_ranker_xgb.pkl"
        active_model = BASE_DIR / "models" / "healing_ranker.pkl"
        if xgb_model.exists():
            # Save current model
            gb_backup = BASE_DIR / "models" / "healing_ranker_gb_backup.pkl"
            if active_model.exists():
                shutil.copy2(active_model, gb_backup)

            # Swap in XGBoost
            shutil.copy2(xgb_model, active_model)
            print("[INFO] Swapped XGBoost model into active slot")

            run_step(
                "STEP 24a - Run ML experiments with XGBoost ranker",
                [python_exe, "run_experiment.py",
                 "--healer-mode", "ml",
                 "--runs", str(args.ablation_runs),
                 "--results-file", "reports/results_ml_xgb.csv"],
            )

            run_step(
                "STEP 24b - Run Hybrid experiments with XGBoost ranker",
                [python_exe, "run_experiment.py",
                 "--healer-mode", "hybrid",
                 "--runs", str(args.ablation_runs),
                 "--results-file", "reports/results_hybrid_xgb.csv"],
            )

            # Restore best model (train_multimodel.py --use-best already selected)
            if gb_backup.exists():
                best_model = BASE_DIR / "models" / "healing_ranker.pkl"
                # train_multimodel.py --use-best already placed the best model
                print("[INFO] Best model is already in the active slot from train_multimodel.py")

        # STEP 25 — Multi-model ablation analysis
        run_step(
            "STEP 25 - Multi-model ablation analysis (GB vs XGBoost)",
            [python_exe, "experiment/generate_multimodel_ablation.py"],
        )

    # ══════════════════════════════════════════════════════════════════
    # RETRAIN FROM ACCUMULATED EVENTS (opt-in via --retrain-from-events)
    # ══════════════════════════════════════════════════════════════════
    if args.retrain_from_events:
        retrain_args = [
            python_exe, "scripts/accumulate_training_data.py",
            "--retrain", "--algorithm", args.ml_algorithm,
        ]
        run_step(
            "STEP 26 - Accumulate training data and retrain model",
            retrain_args,
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated outputs:")
    print("  reports/results.csv")
    print("  reports/mutation_report.csv")
    print("  reports/locator_healing_report.csv")
    print("  reports/mutation_healing_summary.csv")
    print("  figures/*.png")

    if args.with_ml:
        print("\nML outputs:")
        print("  data/training/healing_training_data.csv")
        print("  models/healing_ranker.pkl")
        print("  models/healing_ranker.metrics.json")
        print("  reports/results_ml.csv")
        print("  reports/results_hybrid.csv")
        print("  reports/ml_ablation_summary.csv")
        print("  reports/ml_ablation_table.tex")
        print("  reports/ml_feature_importance.csv")
        print("  figures/ml_ablation_comparison.png")
        print("  figures/ml_ablation_per_version.png")
        print("  figures/ml_score_distribution.png")
        print("  figures/ml_feature_importance.png")

    if args.with_multimodel:
        print("\nMulti-model outputs:")
        print("  models/healing_ranker_gb.pkl")
        print("  models/healing_ranker_xgb.pkl")
        print("  models/model_comparison.json")
        print("  reports/model_comparison_table.tex")
        print("  reports/multimodel_ablation_table.tex")
        print("  figures/model_comparison.png")
        print("  figures/multimodel_ablation.png")
        print("  figures/multimodel_per_version.png")
        if args.with_hp_sweep:
            print("  models/sweep_results.csv")
            print("  models/best_model_summary.json")
            print("  reports/hp_sweep_table.tex")
            print("  figures/hp_sweep_heatmap.png")


if __name__ == "__main__":
    main()