"""Rule-based baseline ranker.

A non-learning, non-scoring fallback used to quantify how much of the
framework's headline recovery rate is attributable to similarity scoring
rather than to naive attribute-fallback. The ranker walks a fixed priority
list of locator attributes and returns the FIRST candidate whose attribute
exactly matches the corresponding attribute on the original element. No
similarity scoring, no threshold, no learning.

Priority (stops at first hit):
    1. element_id  (exact match)
    2. name        (exact match)
    3. placeholder (exact match)
    4. aria_label  (exact match)
    5. class_name  (exact first-class match) + tag match
    6. tag match   (first candidate with same tag)

If no rule fires, returns None (healing reported as failure).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RuleBasedRanker:

    def __init__(self, threshold: float = 0.0):
        # threshold retained for API parity with HeuristicRanker
        self.threshold = threshold
        self.weights = {"rule_based": 1.0}

    def best_candidate(self, original, candidates) -> Optional[Dict[str, Any]]:

        if not candidates:
            return None

        def _pick(attr: str):
            orig_val = getattr(original, attr, None)
            if not orig_val:
                return None
            for c in candidates:
                if getattr(c, attr, None) == orig_val:
                    return c
            return None

        # Priority 1..4: exact attribute match
        for attr in ("element_id", "name", "placeholder", "aria_label"):
            hit = _pick(attr)
            if hit is not None:
                return {
                    "candidate": hit,
                    "scores": self._breakdown(f"attr:{attr}", 1.0),
                }

        # Priority 5: first-class + tag match
        orig_classes = (getattr(original, "class_name", "") or "").split()
        if orig_classes:
            first_class = orig_classes[0]
            for c in candidates:
                c_classes = (getattr(c, "class_name", "") or "").split()
                if first_class in c_classes and c.tag == original.tag:
                    return {
                        "candidate": c,
                        "scores": self._breakdown("attr:class+tag", 0.8),
                    }

        # Priority 6: any candidate with same tag
        for c in candidates:
            if c.tag == original.tag:
                return {
                    "candidate": c,
                    "scores": self._breakdown("attr:tag", 0.5),
                }

        return None

    @staticmethod
    def _breakdown(rule: str, total: float) -> Dict[str, Any]:
        return {
            "tag_match": 1 if rule != "attr:tag" else 1,
            "text_similarity": 0.0,
            "attribute_similarity": 1.0 if rule.startswith("attr:") and "class" not in rule and rule != "attr:tag" else 0.0,
            "class_similarity": 1.0 if "class" in rule else 0.0,
            "parent_similarity": 0.0,
            "depth_similarity": 0.0,
            "total_score": total,
            "rule_fired": rule,
        }
