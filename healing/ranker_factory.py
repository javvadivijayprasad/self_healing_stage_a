"""Factory for constructing a ranker given a healer mode.

Supported modes:
  - heuristic   : similarity-weighted HeuristicRanker (default)
  - rule_based  : attribute-fallback baseline (no scoring)
  - random      : random same-tag candidate (lower-bound baseline)
  - ml          : trained ML classifier (Gradient Boosting / XGBoost)
  - hybrid      : ensemble of heuristic + ML (α-weighted)
  - none        : disable healing entirely (baseline-only)

Stage B (Paper 2): The ml and hybrid modes are now backed by trained
model artifacts in models/healing_ranker.pkl. If the artifact is missing,
the MLRanker and HybridRanker gracefully fall back to heuristic scoring
internally, so the pipeline stays backward-compatible.
"""

from __future__ import annotations

from typing import Any

from healing.heuristic_ranker import HeuristicRanker
from healing.rule_based_ranker import RuleBasedRanker
from healing.random_ranker import RandomRanker
from healing.ml_ranker import MLRanker
from healing.hybrid_ranker import HybridRanker


VALID_MODES = ("heuristic", "rule_based", "random", "ml", "hybrid", "none")


def create_ranker(mode: str, threshold: float = 0.45, seed: int = 0) -> Any:
    """Create a ranker instance for the given mode.

    Parameters
    ----------
    mode : str
        One of VALID_MODES.
    threshold : float
        Score/probability threshold for accepting a candidate.
        - heuristic/rule_based: similarity score threshold (default 0.45)
        - ml/hybrid: probability threshold (default 0.50, adjusted internally)
    seed : int
        RNG seed for the random ranker.
    """
    mode = (mode or "heuristic").lower().strip()

    if mode == "heuristic":
        return HeuristicRanker(threshold=threshold)

    if mode == "rule_based":
        return RuleBasedRanker()

    if mode == "random":
        return RandomRanker(seed=seed)

    if mode == "ml":
        # ML ranker uses slightly higher default threshold (probability)
        ml_threshold = max(threshold, 0.50)
        return MLRanker(threshold=ml_threshold)

    if mode == "hybrid":
        # Hybrid ensembles heuristic (40%) + ML (60%)
        hybrid_threshold = max(threshold, 0.45)
        return HybridRanker(threshold=hybrid_threshold, alpha=0.40)

    if mode == "none":
        return _NullRanker()

    raise ValueError(f"Unknown healer mode: {mode!r}. "
                     f"Expected one of {VALID_MODES}")


class _NullRanker:
    """Ranker that never returns a candidate (disables healing)."""

    threshold = 1.01

    def best_candidate(self, original, candidates):
        return None
