#!/usr/bin/env python3
"""
Heal Event Logger — captures every healing decision with full feature vectors.

══════════════════════════════════════════════════════════════════════════════
Paper 2: Self-Healing Test Automation via ML-Enhanced Locator Recovery
         Vijay P. Javvadi — 2024-2026
══════════════════════════════════════════════════════════════════════════════

Every time the self-healing engine encounters a broken locator, this logger
records the FULL context: original element metadata, all candidate elements,
the 13-feature vector for each candidate, the ranking decision, and the
outcome (healed / failed). This produces a continuously growing labeled
dataset that feeds future model retraining.

Logged events are appended to:
  data/heal_events/heal_events.csv    (one row per candidate evaluated)
  data/heal_events/heal_decisions.csv (one row per healing decision)

The candidate-level CSV is directly usable as ML training data:
  - label=1 for the candidate that was selected AND successfully healed
  - label=0 for all other candidates (not selected, or selected but failed)

Usage:
  # Integrated automatically into SelfHealEngine (no manual calls needed)
  # Or standalone for batch logging:
  logger = HealEventLogger()
  logger.log_event(event)
  logger.flush()
"""
from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from healing.utils import ensure_dir


# Feature columns — must match extract_training_data.py and ml_ranker.py
FEATURE_COLUMNS = [
    "tag_match", "text_similarity", "attribute_similarity", "class_similarity",
    "parent_similarity", "depth_similarity",
    "id_similarity", "name_similarity", "placeholder_similarity",
    "aria_label_similarity", "type_attr_match", "sibling_count_ratio",
    "multi_attr_match_count",
]

DECISION_COLUMNS = [
    "timestamp", "test_case", "page_name", "element_name",
    "original_locator", "locator_type", "ui_version",
    "n_candidates", "ranker_mode", "selected_candidate_idx",
    "selected_score", "outcome",  # heal_success | heal_failed | no_candidate
    "healed_locator", "original_tag",
    "score_breakdown_json",
]

CANDIDATE_COLUMNS = [
    "timestamp", "event_id", "test_case", "page_name", "element_name",
    "ui_version", "candidate_idx", "candidate_tag", "candidate_id",
    "candidate_css_path",
] + FEATURE_COLUMNS + [
    "score", "rank", "was_selected", "label",
]


@dataclass
class CandidateRecord:
    """One evaluated candidate with features and outcome."""
    candidate_idx: int = 0
    candidate_tag: str = ""
    candidate_id: str = ""
    candidate_css_path: str = ""
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    was_selected: bool = False
    label: int = 0  # 1 = correct (selected + healed), 0 = incorrect


@dataclass
class HealEvent:
    """Complete healing decision with all candidates and context."""
    timestamp: str = ""
    event_id: str = ""
    test_case: str = ""
    page_name: str = ""
    element_name: str = ""
    original_locator: str = ""
    locator_type: str = ""
    ui_version: str = ""
    original_tag: str = ""
    n_candidates: int = 0
    ranker_mode: str = "heuristic"
    selected_candidate_idx: int = -1
    selected_score: float = 0.0
    outcome: str = ""  # heal_success | heal_failed | heal_failed_resolution | no_candidate
    healed_locator: str = ""
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    candidates: List[CandidateRecord] = field(default_factory=list)


class HealEventLogger:
    """Thread-safe logger that appends healing events to CSV files."""

    def __init__(
        self,
        output_dir: str = "data/heal_events",
        flush_interval: int = 10,  # flush every N events
    ):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

        self.decisions_path = self.output_dir / "heal_decisions.csv"
        self.candidates_path = self.output_dir / "heal_events.csv"

        self._decision_buffer: List[Dict[str, Any]] = []
        self._candidate_buffer: List[Dict[str, Any]] = []
        self._flush_interval = flush_interval
        self._event_count = 0
        self._lock = threading.Lock()

        # Write headers if files don't exist
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        """Write CSV headers if files don't exist or are empty."""
        if not self.decisions_path.exists() or self.decisions_path.stat().st_size == 0:
            with open(self.decisions_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(DECISION_COLUMNS)

        if not self.candidates_path.exists() or self.candidates_path.stat().st_size == 0:
            with open(self.candidates_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CANDIDATE_COLUMNS)

    def log_event(self, event: HealEvent) -> None:
        """Log a complete healing event (thread-safe, buffered)."""
        with self._lock:
            # Decision row
            decision_row = {
                "timestamp": event.timestamp,
                "test_case": event.test_case,
                "page_name": event.page_name,
                "element_name": event.element_name,
                "original_locator": event.original_locator,
                "locator_type": event.locator_type,
                "ui_version": event.ui_version,
                "n_candidates": event.n_candidates,
                "ranker_mode": event.ranker_mode,
                "selected_candidate_idx": event.selected_candidate_idx,
                "selected_score": round(event.selected_score, 6),
                "outcome": event.outcome,
                "healed_locator": event.healed_locator,
                "original_tag": event.original_tag,
                "score_breakdown_json": json.dumps(event.score_breakdown),
            }
            self._decision_buffer.append(decision_row)

            # Candidate rows (one per candidate evaluated)
            for cand in event.candidates:
                cand_row = {
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "test_case": event.test_case,
                    "page_name": event.page_name,
                    "element_name": event.element_name,
                    "ui_version": event.ui_version,
                    "candidate_idx": cand.candidate_idx,
                    "candidate_tag": cand.candidate_tag,
                    "candidate_id": cand.candidate_id,
                    "candidate_css_path": cand.candidate_css_path,
                    "score": round(cand.score, 6),
                    "rank": cand.rank,
                    "was_selected": int(cand.was_selected),
                    "label": cand.label,
                }
                # Add feature columns
                for feat in FEATURE_COLUMNS:
                    cand_row[feat] = round(cand.features.get(feat, 0.0), 6)

                self._candidate_buffer.append(cand_row)

            self._event_count += 1

            if self._event_count % self._flush_interval == 0:
                self._flush_unlocked()

    def flush(self) -> None:
        """Force flush all buffered events to disk."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Flush without acquiring lock (caller must hold lock)."""
        if self._decision_buffer:
            with open(self.decisions_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
                writer.writerows(self._decision_buffer)
            self._decision_buffer.clear()

        if self._candidate_buffer:
            with open(self.candidates_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
                writer.writerows(self._candidate_buffer)
            self._candidate_buffer.clear()

    def stats(self) -> Dict[str, Any]:
        """Return current logger statistics."""
        n_decisions = 0
        n_candidates = 0
        n_positive = 0

        if self.decisions_path.exists():
            with open(self.decisions_path, "r", encoding="utf-8") as f:
                n_decisions = sum(1 for _ in f) - 1  # minus header

        if self.candidates_path.exists():
            with open(self.candidates_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n_candidates += 1
                    if row.get("label") == "1":
                        n_positive += 1

        return {
            "total_decisions": n_decisions,
            "total_candidates": n_candidates,
            "positive_labels": n_positive,
            "negative_labels": n_candidates - n_positive,
            "decisions_path": str(self.decisions_path),
            "candidates_path": str(self.candidates_path),
        }
