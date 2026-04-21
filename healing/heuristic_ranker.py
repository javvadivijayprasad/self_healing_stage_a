from dataclasses import asdict
from typing import List, Dict, Any
import difflib


class HeuristicRanker:

    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

        # weights must sum to 1.0
        self.weights = {
            "tag_match": 0.15,
            "text_similarity": 0.20,
            "attribute_similarity": 0.25,
            "class_similarity": 0.15,
            "parent_similarity": 0.15,
            "depth_similarity": 0.10
        }

    def best_candidate(self, original, candidates):

        best_candidate = None
        best_score = 0
        best_breakdown = None

        for candidate in candidates:

            scores = self._calculate_scores(original, candidate)

            total_score = sum(
                scores[key] * self.weights[key]
                for key in self.weights
            )

            scores["total_score"] = total_score

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
                best_breakdown = scores

        if best_score < self.threshold:
            return None

        return {
            "candidate": best_candidate,
            "scores": best_breakdown
        }

    def _calculate_scores(self, original, candidate):

        scores = {}

        # ---------- tag match ----------
        scores["tag_match"] = 1 if original.tag == candidate.tag else 0

        # ---------- text similarity ----------
        scores["text_similarity"] = self._string_similarity(
            original.text,
            candidate.text
        )

        # ---------- attribute similarity ----------
        attr_score = max(
            self._string_similarity(original.element_id, candidate.element_id),
            self._string_similarity(original.name, candidate.name),
            self._string_similarity(original.placeholder, candidate.placeholder),
            self._string_similarity(original.aria_label, candidate.aria_label),
        )

        scores["attribute_similarity"] = attr_score

        # ---------- class similarity ----------
        scores["class_similarity"] = self._string_similarity(
            original.class_name,
            candidate.class_name
        )

        # ---------- parent tag similarity ----------
        scores["parent_similarity"] = (
            1 if original.parent_tag == candidate.parent_tag else 0
        )

        # ---------- DOM depth similarity ----------
        if original.dom_depth and candidate.dom_depth:
            depth_diff = abs(original.dom_depth - candidate.dom_depth)
            scores["depth_similarity"] = max(0, 1 - (depth_diff / 10))
        else:
            scores["depth_similarity"] = 0

        return scores

    def _string_similarity(self, a, b):

        if not a or not b:
            return 0

        a = str(a).lower()
        b = str(b).lower()

        return difflib.SequenceMatcher(None, a, b).ratio()