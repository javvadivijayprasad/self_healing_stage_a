"""Random-candidate baseline ranker.

Lower-bound sanity baseline. Picks a random candidate that matches the
original element's tag (to avoid picking, e.g., a <div> when the test
wanted an <input>). Used to confirm that the heuristic/ML rankers add
signal beyond chance.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class RandomRanker:

    def __init__(self, threshold: float = 0.0, seed: Optional[int] = None):
        self.threshold = threshold
        self.rng = random.Random(seed)

    def best_candidate(self, original, candidates) -> Optional[Dict[str, Any]]:

        if not candidates:
            return None

        same_tag = [c for c in candidates if c.tag == original.tag]
        pool = same_tag if same_tag else candidates

        if not pool:
            return None

        pick = self.rng.choice(pool)

        return {
            "candidate": pick,
            "scores": {
                "tag_match": 1 if pick.tag == original.tag else 0,
                "text_similarity": 0.0,
                "attribute_similarity": 0.0,
                "class_similarity": 0.0,
                "parent_similarity": 0.0,
                "depth_similarity": 0.0,
                "total_score": 0.0,
                "rule_fired": "random",
            },
        }
