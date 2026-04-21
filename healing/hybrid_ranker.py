"""Hybrid ranker — ensembles heuristic scoring with ML prediction.

Paper 2, Stage B: the hybrid strategy combines the interpretable,
hand-tuned heuristic weights with the learned feature interactions
from the ML classifier. This addresses a key research question:
does ML improve over the heuristic, and does ensembling improve
over ML alone?

Ensemble formula:
    final_score = α × heuristic_score + (1 − α) × ml_probability

Default α = 0.40 (heuristic contributes 40%, ML contributes 60%).
This weighting was chosen because the ML model has access to more
features (13 vs 6) and can learn non-linear interactions.

Interface contract (same as HeuristicRanker / MLRanker):
    best_candidate(original, candidates) → Optional[Dict]
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

from healing.ml_ranker import MLRanker, _compute_features, FEATURE_COLUMNS, MODEL_PATH


class HybridRanker:
    """Ensemble ranker combining heuristic weights with ML probability.

    Parameters
    ----------
    threshold : float
        Minimum ensemble score to accept a candidate (default 0.50).
    alpha : float
        Weight given to the heuristic component (default 0.40).
        ML component receives weight (1 - alpha).
    model_path : str | None
        Path to trained ML model. Defaults to models/healing_ranker.pkl.
    """

    # Heuristic weights (identical to HeuristicRanker)
    HEURISTIC_WEIGHTS = {
        "tag_match": 0.15,
        "text_similarity": 0.20,
        "attribute_similarity": 0.25,
        "class_similarity": 0.15,
        "parent_similarity": 0.15,
        "depth_similarity": 0.10,
    }

    def __init__(
        self,
        threshold: float = 0.50,
        alpha: float = 0.40,
        model_path: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.alpha = alpha

        # Load the ML model (will fallback to heuristic-only if unavailable)
        self._ml_ranker = MLRanker(
            threshold=0.0,  # We handle thresholding ourselves
            model_path=model_path,
        )
        self._ml_available = (
            self._ml_ranker._pipeline is not None
            and not self._ml_ranker._fallback
        )

        if not self._ml_available:
            print(f"[HybridRanker] ML model unavailable; "
                  f"operating in heuristic-only mode (α=1.0).")

    def best_candidate(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Score candidates using the heuristic + ML ensemble.

        If the ML model is unavailable, falls back to pure heuristic
        scoring (equivalent to HeuristicRanker).
        """
        if not candidates:
            return None

        if not self._ml_available:
            return self._heuristic_only(original, candidates)

        return self._ensemble_score(original, candidates)

    def _ensemble_score(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Score each candidate with both heuristic and ML, then ensemble."""
        import numpy as np

        # Compute features for all candidates
        feature_data = []
        for cand in candidates:
            feats = _compute_features(original, cand)
            row = [feats[col] for col in FEATURE_COLUMNS]
            feature_data.append((cand, feats, row))

        if not feature_data:
            return None

        # Batch ML prediction
        X = np.array([r[2] for r in feature_data], dtype=float)
        try:
            ml_probas = self._ml_ranker._pipeline.predict_proba(X)[:, 1]
        except Exception as exc:
            print(f"[HybridRanker] ML prediction failed: {exc}; "
                  f"falling back to heuristic-only.")
            return self._heuristic_only(original, candidates)

        best_candidate = None
        best_score = 0.0
        best_scores: Dict[str, Any] = {}

        for i, (cand, feats, _) in enumerate(feature_data):
            # Heuristic component
            heuristic_score = sum(
                feats.get(k, 0.0) * w
                for k, w in self.HEURISTIC_WEIGHTS.items()
            )

            # ML component
            ml_prob = float(ml_probas[i])

            # Ensemble
            ensemble_score = self.alpha * heuristic_score + (1.0 - self.alpha) * ml_prob

            if ensemble_score > best_score:
                best_score = ensemble_score
                best_candidate = cand
                best_scores = {
                    **feats,
                    "heuristic_score": round(heuristic_score, 4),
                    "ml_probability": round(ml_prob, 4),
                    "alpha": self.alpha,
                    "total_score": round(ensemble_score, 4),
                    "ranker_mode": "hybrid",
                }

        if best_score < self.threshold:
            return None

        return {
            "candidate": best_candidate,
            "scores": best_scores,
        }

    def _heuristic_only(
        self,
        original: Any,
        candidates: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Pure heuristic fallback (no ML component)."""
        best_candidate = None
        best_score = 0.0
        best_feats: Dict[str, float] = {}

        for cand in candidates:
            feats = _compute_features(original, cand)
            total = sum(
                feats.get(k, 0.0) * w
                for k, w in self.HEURISTIC_WEIGHTS.items()
            )
            if total > best_score:
                best_score = total
                best_candidate = cand
                best_feats = feats

        if best_score < self.threshold:
            return None

        return {
            "candidate": best_candidate,
            "scores": {
                **best_feats,
                "heuristic_score": round(best_score, 4),
                "ml_probability": None,
                "alpha": 1.0,
                "total_score": round(best_score, 4),
                "ranker_mode": "hybrid_fallback_heuristic",
            },
        }
