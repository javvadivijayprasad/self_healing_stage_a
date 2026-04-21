#!/usr/bin/env python3
"""
Extract ML training data from experiment results.

══════════════════════════════════════════════════════════════════════════════
Paper 2 – Stage B: ML-Enhanced Self-Healing Locator Recovery
Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Reads the canonical `reports/results.csv` (heuristic-mode experiment output)
and the locator metadata in `data/processed/locator_store/` to construct a
labelled training dataset suitable for XGBoost / Gradient Boosting.

Each row represents one (LocatorMetadata, CandidateElement) pair evaluated
during healing, with:
  - 13 similarity features (6 heuristic + 7 ML-only)
  - Binary label: 1 = this candidate was selected and healing succeeded,
                  0 = this candidate was not selected OR healing failed

Pipeline:
  1. Parse results.csv, extract JSON details per row
  2. For each healing event, reconstruct original metadata + all candidates
  3. Compute 13-feature vector for each (original, candidate) pair
  4. Label positive (heal_success + selected) vs negative
  5. Write to data/training/healing_training_data.csv

Usage:
  python scripts/extract_training_data.py
  python scripts/extract_training_data.py --results reports/results.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is on path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from healing.utils import ensure_dir


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    # --- 6 heuristic features (match HeuristicRanker) ---
    "tag_match",
    "text_similarity",
    "attribute_similarity",
    "class_similarity",
    "parent_similarity",
    "depth_similarity",
    # --- 7 ML-only features ---
    "id_similarity",
    "name_similarity",
    "placeholder_similarity",
    "aria_label_similarity",
    "type_attr_match",
    "sibling_count_ratio",
    "multi_attr_match_count",
]


def _str_sim(a: str, b: str) -> float:
    """Case-insensitive SequenceMatcher ratio (matches HeuristicRanker)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def compute_features(original: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute the 13-feature vector for an (original, candidate) pair.

    Parameters
    ----------
    original : dict
        LocatorMetadata fields (from locator_store JSON or step_log).
    candidate : dict
        CandidateElement fields (from healed_candidate in step_log).

    Returns
    -------
    dict
        Feature name → float value.
    """
    features: Dict[str, float] = {}

    # ── Heuristic features (reproduce HeuristicRanker exactly) ────────────
    o_tag = (original.get("tag") or "").lower()
    c_tag = (candidate.get("tag") or "").lower()
    features["tag_match"] = 1.0 if o_tag == c_tag else 0.0

    features["text_similarity"] = _str_sim(
        original.get("text", ""), candidate.get("text", "")
    )

    # attribute_similarity = max across id, name, placeholder, aria_label
    id_sim = _str_sim(original.get("element_id", ""), candidate.get("element_id", ""))
    name_sim = _str_sim(original.get("name", ""), candidate.get("name", ""))
    plc_sim = _str_sim(original.get("placeholder", ""), candidate.get("placeholder", ""))
    aria_sim = _str_sim(original.get("aria_label", ""), candidate.get("aria_label", ""))
    features["attribute_similarity"] = max(id_sim, name_sim, plc_sim, aria_sim)

    features["class_similarity"] = _str_sim(
        original.get("class_name", ""), candidate.get("class_name", "")
    )

    o_parent = (original.get("parent_tag") or "").lower()
    c_parent = (candidate.get("parent_tag") or "").lower()
    features["parent_similarity"] = 1.0 if o_parent and o_parent == c_parent else 0.0

    o_depth = original.get("dom_depth", 0) or 0
    c_depth = candidate.get("dom_depth", 0) or 0
    if o_depth and c_depth:
        features["depth_similarity"] = max(0.0, 1.0 - abs(o_depth - c_depth) / 10.0)
    else:
        features["depth_similarity"] = 0.0

    # ── ML-only features ──────────────────────────────────────────────────
    features["id_similarity"] = id_sim
    features["name_similarity"] = name_sim
    features["placeholder_similarity"] = plc_sim
    features["aria_label_similarity"] = aria_sim

    # type_attr match (input type="text" vs type="password" etc.)
    o_type = (original.get("type_attr") or original.get("locator_type") or "").lower()
    c_type = (candidate.get("type_attr") or "").lower()
    features["type_attr_match"] = 1.0 if o_type and o_type == c_type else 0.0

    # sibling_count_ratio: normalised sibling count (structural density)
    c_siblings = candidate.get("sibling_count", 0) or 0
    features["sibling_count_ratio"] = min(c_siblings / 20.0, 1.0)

    # multi_attr_match_count: how many attributes match simultaneously
    match_count = 0
    for attr in ("element_id", "name", "placeholder", "aria_label"):
        o_val = original.get(attr, "")
        c_val = candidate.get(attr, "")
        if o_val and c_val and o_val.strip().lower() == c_val.strip().lower():
            match_count += 1
    features["multi_attr_match_count"] = float(match_count)

    return features


# ═══════════════════════════════════════════════════════════════════════════
# Data extraction from results.csv
# ═══════════════════════════════════════════════════════════════════════════

def _parse_details(raw_details: str) -> Optional[Dict[str, Any]]:
    """Safely parse the JSON details column."""
    try:
        return json.loads(raw_details)
    except (json.JSONDecodeError, TypeError):
        return None


def _load_locator_metadata(store_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all locator metadata JSONs into a lookup dict.

    Key: "{test_case}__{element_name}" → dict of LocatorMetadata fields.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    if not store_dir.exists():
        return lookup
    for f in store_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            key = f"{data.get('test_case', '')}__{data.get('element_name', '')}"
            lookup[key] = data
        except Exception:
            continue
    return lookup


def extract_training_rows(
    results_path: Path,
    store_dir: Path,
) -> List[Dict[str, Any]]:
    """Extract labelled (original, candidate, features, label) rows."""

    metadata_lookup = _load_locator_metadata(store_dir)
    training_rows: List[Dict[str, Any]] = []

    with open(results_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            details = _parse_details(row.get("details", ""))
            if not details or "step_logs" not in details:
                continue

            ui_version = row.get("ui_version", "")
            test_name = row.get("test_name", "")

            for step in details["step_logs"]:
                status = step.get("status", "")

                # We only care about healing attempts (not baseline successes)
                if status not in ("heal_success", "heal_failed", "heal_failed_resolution"):
                    continue

                test_case = step.get("test_case", "")
                element_name = step.get("element_name", "")
                lookup_key = f"{test_case}__{element_name}"

                # Get original metadata
                original = metadata_lookup.get(lookup_key)
                if not original:
                    continue

                # Get the selected candidate (if healing succeeded)
                healed_candidate = step.get("healed_candidate")
                is_positive = status == "heal_success" and healed_candidate is not None

                if is_positive:
                    # Positive example: the candidate that was selected and worked
                    features = compute_features(original, healed_candidate)
                    training_rows.append({
                        **features,
                        "label": 1,
                        "ui_version": ui_version,
                        "test_name": test_name,
                        "element_name": element_name,
                        "status": status,
                    })

                    # Generate negative examples from the score breakdown
                    # We create synthetic negatives by perturbing the positive
                    _add_synthetic_negatives(
                        training_rows, original, healed_candidate,
                        ui_version, test_name, element_name,
                    )

                elif status in ("heal_failed", "heal_failed_resolution"):
                    # If a candidate was ranked but resolution failed,
                    # it's a negative (wrong candidate or unresolvable)
                    if healed_candidate:
                        features = compute_features(original, healed_candidate)
                        training_rows.append({
                            **features,
                            "label": 0,
                            "ui_version": ui_version,
                            "test_name": test_name,
                            "element_name": element_name,
                            "status": status,
                        })

    return training_rows


def _add_synthetic_negatives(
    rows: List[Dict[str, Any]],
    original: Dict[str, Any],
    positive_candidate: Dict[str, Any],
    ui_version: str,
    test_name: str,
    element_name: str,
) -> None:
    """Create synthetic negative examples by degrading the positive candidate.

    Since we don't have the full candidate list stored in logs, we create
    plausible negatives by zeroing out key attributes of the positive candidate.
    This teaches the model which features matter for correct identification.
    """
    degradations = [
        # Wrong ID
        {**positive_candidate, "element_id": "", "name": ""},
        # Wrong text
        {**positive_candidate, "text": "other-text", "placeholder": "other"},
        # Wrong tag
        {**positive_candidate, "tag": "div" if positive_candidate.get("tag") != "div" else "span"},
        # Wrong parent
        {**positive_candidate, "parent_tag": "section", "dom_depth": (positive_candidate.get("dom_depth", 5) or 5) + 3},
    ]

    for degraded in degradations:
        features = compute_features(original, degraded)
        rows.append({
            **features,
            "label": 0,
            "ui_version": ui_version,
            "test_name": test_name,
            "element_name": element_name,
            "status": "synthetic_negative",
        })


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ML training data from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", default=str(PROJECT_ROOT / "reports" / "results.csv"),
        help="Path to results.csv from heuristic experiment run.",
    )
    parser.add_argument(
        "--store-dir", default=str(PROJECT_ROOT / "data" / "processed" / "locator_store"),
        help="Path to locator_store directory with baseline metadata JSONs.",
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "training" / "healing_training_data.csv"),
        help="Output CSV path for training data.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    store_dir = Path(args.store_dir)
    output_path = Path(args.output)

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run the heuristic experiment first: python run_experiment.py --healer-mode heuristic")
        sys.exit(1)

    if not store_dir.exists():
        print(f"ERROR: Locator store not found: {store_dir}")
        print("Run bootstrap first: python run_experiment.py")
        sys.exit(1)

    print(f"[EXTRACT] Reading results from {results_path}")
    print(f"[EXTRACT] Loading metadata from {store_dir}")

    rows = extract_training_rows(results_path, store_dir)

    if not rows:
        print("ERROR: No training rows extracted. Check results.csv format.")
        sys.exit(1)

    # Write CSV
    ensure_dir(output_path.parent)
    fieldnames = FEATURE_COLUMNS + ["label", "ui_version", "test_name", "element_name", "status"]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)

    print(f"\n{'─'*50}")
    print(f"  Training data extracted successfully")
    print(f"  Total rows  : {len(rows):,}")
    print(f"  Positive (1): {n_pos:,} ({n_pos/len(rows)*100:.1f}%)")
    print(f"  Negative (0): {n_neg:,} ({n_neg/len(rows)*100:.1f}%)")
    print(f"  Features    : {len(FEATURE_COLUMNS)}")
    print(f"  Output      : {output_path}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
