#!/usr/bin/env python3
"""
Generate the three-panel feature-importance comparison figure for paper2
§7.2 and §7.3.

Produces ``figures/feature_importance_comparison.png`` showing:
  (a) Phase 1 — original Gradient Boosting (13 features, n=6,200 train)
  (b) Phase 2 — augmented Gradient Boosting (13 features, n=7,400 train)
  (c) Phase 3 — augmented Gradient Boosting (17 features, n=9,800 train)

The Phase 1 importances are read from
``models/healing_ranker_gb.metrics.json`` if available; otherwise fallback
constants are used. Phase 2 and Phase 3 importances are hard-coded here
from the published training runs (the training script prints them but
does not persist them to JSON).

Usage:
  python experiment/generate_feature_importance_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

ORIGINAL_METRICS = (
    PROJECT_ROOT / "models" / "healing_ranker_gb.metrics.json"
)


# ── Phase 1 fallback (if the metrics.json is missing) ───────────────────────
PHASE_1 = {
    "name_similarity":            0.0000,
    "text_similarity":            0.3796,
    "tag_match":                  0.0871,
    "id_similarity":              0.2063,
    "parent_similarity":          0.1420,
    "placeholder_similarity":     0.1554,
    "attribute_similarity":       0.0196,
    "multi_attr_match_count":     0.0000,
    "depth_similarity":           0.0060,
    "sibling_count_ratio":        0.0041,
    "class_similarity":           0.0000,
    "aria_label_similarity":      0.0000,
    "type_attr_match":            0.0000,
}

# ── Phase 2 (paper2 §7.2) — name-heavy augmentation ─────────────────────────
PHASE_2 = {
    "name_similarity":            0.2936,
    "text_similarity":            0.2282,
    "tag_match":                  0.1405,
    "id_similarity":              0.1376,
    "parent_similarity":          0.1142,
    "placeholder_similarity":     0.0496,
    "attribute_similarity":       0.0169,
    "multi_attr_match_count":     0.0135,
    "depth_similarity":           0.0032,
    "sibling_count_ratio":        0.0026,
    "class_similarity":           0.0001,
    "aria_label_similarity":      0.0000,
    "type_attr_match":            0.0000,
}

# ── Phase 3 (paper2 §7.3) — candidate-intrinsic schema ──────────────────────
PHASE_3 = {
    "type_attr_match":            0.2669,
    "text_similarity":            0.1084,
    "parent_similarity":          0.1058,
    "tag_match":                  0.1006,
    "candidate_attr_richness":    0.0998,
    "name_similarity":            0.0678,
    "id_similarity":              0.0651,
    "attribute_similarity":       0.0533,
    "candidate_text_actionable":  0.0336,
    "depth_similarity":           0.0288,
    "candidate_in_form":          0.0281,
    "placeholder_similarity":     0.0206,
    "multi_attr_match_count":     0.0184,
    "class_similarity":           0.0020,
    "sibling_count_ratio":        0.0007,
    "aria_label_similarity":      0.0000,
    "original_attr_count":        0.0000,
}


def _load_phase1() -> dict:
    if ORIGINAL_METRICS.exists():
        with ORIGINAL_METRICS.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fi = data.get("feature_importance")
        if fi:
            return fi
    return PHASE_1


def _draw_panel(ax, importances: dict, title: str, color: str, all_keys) -> None:
    # Show every feature on every panel (zero-padded for missing) so the
    # rows align across panels and reviewers can read across.
    items = [(k, importances.get(k, 0.0)) for k in all_keys]
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    y = np.arange(len(labels))
    ax.barh(y, values, color=color, edgecolor="#333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.40)
    ax.set_xlabel("Feature importance", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for i, v in enumerate(values):
        if v > 0.001:
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6.5)


def main() -> None:
    phase1 = _load_phase1()

    # Union of all feature names so all three panels show identical rows
    all_keys = sorted(set(phase1) | set(PHASE_2) | set(PHASE_3))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.5), sharey=False)
    _draw_panel(axes[0], phase1,  "Phase 1 — Original (13 feats, 6,200 train)",        "#4F86C6", all_keys)
    _draw_panel(axes[1], PHASE_2, "Phase 2 — Name-heavy aug (13 feats, 7,400 train)", "#3CB371", all_keys)
    _draw_panel(axes[2], PHASE_3, "Phase 3 — Cand-intrinsic (17 feats, 9,800 train)", "#E07B39", all_keys)
    fig.suptitle(
        "Feature Importance: Three Generations of the GB Healing Ranker",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIGURE_DIR / "feature_importance_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
