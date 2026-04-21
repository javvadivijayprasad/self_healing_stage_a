"""ML-based ranker — Paper 2, Stage B.

Uses a trained Gradient Boosting / XGBoost classifier to score
(LocatorMetadata, CandidateElement) pairs and select the best candidate.

The model is loaded from `models/healing_ranker.pkl` at construction time.
If the model artifact is missing, the ranker falls back to the heuristic
scorer and logs a warning — keeping the experiment pipeline backward-compatible.

Interface contract (same as HeuristicRanker):
    best_candidate(original, candidates) → Optional[Dict]
      - original:   LocatorMetadata instance (dataclass)
      - candidates:  List[CandidateElement] instances
      - returns:     {"candidate": CandidateElement, "scores": {...}} or None
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Feature computation (shared with training pipeline) ──────────────────
# We inline the feature computation here rather than importing from scripts/
# to keep the healing package self-contained. The feature logic is identical
# to scripts/extract_training_data.py:compute_features().

FEATURE_COLUMNS = [
    "tag_match",
    "text_similarity",
    "attribute_similarity",
    "class_similarity",
    "parent_similarity",
    "depth_similarity",
    "id_similarity",
    "name_similarity",
    "placeholder_similarity",
    "aria_label_similarity",
    "type_attr_match",
    "sibling_count_ratio",
    "multi_attr_match_count",
]

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "healing_ranker.pkl"


def _str_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _compute_features(original: Any, candidate: Any) -> Dict[str, float]:
    """Compute 13-feature vector from LocatorMetadata + CandidateElement.

    Accepts both dataclass instances and plain dicts so the same code
    works at training time (dicts from JSON) and runtime (dataclasses).
    """
    def _g(obj: Any, attr: str, default: Any = "") -> Any:
        """Get attribute from dataclass or dict."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    features: Dict[str, float] = {}

    o_tag = (_g(original, "tag") or "").lower()
    c_tag = (_g(candidate, "tag") or "").lower()
    features["tag_match"] = 1.0 if o_tag == c_tag else 0.0

    features["text_similarity"] = _str_sim(
        _g(original, "text", ""), _g(candidate, "text", "")
    )

    id_sim = _str_sim(_g(original, "element_id", ""), _g(candidate, "element_id", ""))
    name_sim = _str_sim(_g(original, "name", ""), _g(candidate, "name", ""))
    plc_sim = _str_sim(_g(original, "placeholder", ""), _g(candidate, "placeholder", ""))
    aria_sim = _str_sim(_g(original, "aria_label", ""), _g(candidate, "aria_label", ""))
    features["attribute_similarity"] = max(id_sim, name_sim, plc_sim, aria_sim)

    features["class_similarity"] = _str_sim(
        _g(original, "class_name", ""), _g(candidate, "class_name", "")
    )

    o_parent = (_g(original, "parent_tag") or "").lower()
    c_parent = (_g(candidate, "parent_tag") or "").lower()
    features["parent_similarity"] = 1.0 if o_parent and o_parent == c_parent else 0.0

    o_depth = _g(original, "dom_depth", 0) or 0
    c_depth = _g(candidate, "dom_depth", 0) or 0
    if o_depth and c_depth:
        features["depth_similarity"] = max(0.0, 1.0 - abs(o_depth - c_depth) / 10.0)
    else:
        features["depth_similarity"] = 0.0

    # ML-only features
    features["id_similarity"] = id_sim
    features["name_similarity"] = name_sim
    features["placeholder_similarity"] = plc_sim
    features["aria_label_similarity"] = aria_sim

    o_type = (_g(original, "type_attr") or _g(original, "locator_type") or "").lower()
    c_type = (_g(candidate, "type_attr") or "").lower()
    features["type_attr_match"] = 1.0 if o_type and o_type == c_type else 0.0

    c_siblings = _g(candidate, "sibling_count", 0) or 0
    features["sibling_count_ratio"] = min(c_siblings / 20.0, 1.0)

    match_count = 0
    for attr in ("element_id", "name", "placeholder", "aria_label"):
        o_val = _g(original, attr, "")
        c_val = _g(candidate, attr, "")
        if o_val and c_val and str(o_val).strip().lower() == str(c_val).strip().lower():
            match_count += 1
    features["multi_attr_match_count"] = float(match_count)

    return features


class MLRanker:
    """Trained ML ranker for self-healing locator recovery.

    Parameters
    ----------
    threshold : float
        Minimum predicted probability to accept a candidate (default 0.50).
    model_path : Path | str | None
        Path to the serialised sklearn/imblearn pipeline. Defaults to
        ``models/healing_ranker.pkl`` relative to the project root.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        model_path: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self._model_path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline = None
        self._fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained pipeline from disk."""
        if not self._model_path.exists():
            print(f"[MLRanker] Model not found at {self._model_path}; "
                  f"will use heuristic fallback scoring.")
            self._fallback = True
            return

        try:
            import joblib
            self._pipeline = joblib.load(self._model_path)
            meta = getattr(self._pipeline, "_governance_meta", {})
            version = meta.get("model_version", "unknown")
            auc = meta.get("auc", "?")
            print(f"[MLRanker] Loaded model: {version} (AUC={auc}) "
                  f"from {self._model_path}")
        except Exception as exc:
            print(f"[MLRanker] Failed to load model: {exc}; "
                  f"will use heuristic fallback scoring.")
            self._fallback = True

    def best_candidate(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Score all candidates and return the best above threshold.

        Parameters
        ----------
        original : LocatorMetadata
            Baseline metadata for the element that failed to locate.
        candidates : list of CandidateElement
            Candidate elements extracted from the live DOM.

        Returns
        -------
        dict or None
            ``{"candidate": CandidateElement, "scores": {...}}`` if a
            candidate exceeds the threshold, else ``None``.
        """
        if not candidates:
            return None

        if self._fallback:
            return self._heuristic_fallback(original, candidates)

        return self._ml_score(original, candidates)

    def _ml_score(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Score candidates using the trained ML pipeline."""
        import numpy as np

        best_candidate = None
        best_prob = 0.0
        best_features: Dict[str, float] = {}

        # Batch: compute features for all candidates at once
        feature_rows = []
        for cand in candidates:
            feats = _compute_features(original, cand)
            row = [feats[col] for col in FEATURE_COLUMNS]
            feature_rows.append((cand, feats, row))

        if not feature_rows:
            return None

        # Stack into numpy array for batch prediction
        X = np.array([r[2] for r in feature_rows], dtype=float)

        try:
            probas = self._pipeline.predict_proba(X)[:, 1]
        except Exception as exc:
            print(f"[MLRanker] Prediction failed: {exc}; falling back to heuristic.")
            return self._heuristic_fallback(original, candidates)

        for i, (cand, feats, _) in enumerate(feature_rows):
            prob = float(probas[i])
            if prob > best_prob:
                best_prob = prob
                best_candidate = cand
                best_features = feats

        if best_prob < self.threshold:
            return None

        scores = {
            **best_features,
            "total_score": best_prob,
            "ml_probability": best_prob,
            "ranker_mode": "ml",
        }

        return {
            "candidate": best_candidate,
            "scores": scores,
        }

    def _heuristic_fallback(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Fallback to weighted heuristic scoring (matches HeuristicRanker)."""
        weights = {
            "tag_match": 0.15,
            "text_similarity": 0.20,
            "attribute_similarity": 0.25,
            "class_similarity": 0.15,
            "parent_similarity": 0.15,
            "depth_similarity": 0.10,
        }

        best_candidate = None
        best_score = 0.0
        best_features: Dict[str, float] = {}

        for cand in candidates:
            feats = _compute_features(original, cand)
            total = sum(feats.get(k, 0.0) * w for k, w in weights.items())
            if total > best_score:
                best_score = total
                best_candidate = cand
                best_features = feats

        if best_score < self.threshold:
            return None

        scores = {
            **best_features,
            "total_score": best_score,
            "ml_probability": None,
            "ranker_mode": "ml_fallback_heuristic",
        }

        return {
            "candidate": best_candidate,
            "scores": scores,
        }
