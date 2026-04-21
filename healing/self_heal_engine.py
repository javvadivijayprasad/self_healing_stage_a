from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from healing.candidate_extractor import CandidateExtractor, CandidateElement
from healing.dom_capture import DOMCapture
from healing.heuristic_ranker import HeuristicRanker
from healing.locator_store import LocatorStore
from healing.utils import ensure_dir, timestamp, truncate

# Heal event logger — captures every healing decision for continuous learning
try:
    from healing.heal_event_logger import (
        HealEventLogger, HealEvent, CandidateRecord,
    )
    _LOGGER_AVAILABLE = True
except ImportError:
    _LOGGER_AVAILABLE = False

# Feature computation — shared with ML ranker and training pipeline
try:
    from healing.ml_ranker import _compute_features
    _FEATURES_AVAILABLE = True
except ImportError:
    _FEATURES_AVAILABLE = False


class SelfHealEngine:

    def __init__(
        self,
        locator_store: Optional[LocatorStore] = None,
        dom_capture: Optional[DOMCapture] = None,
        ranker: Optional[HeuristicRanker] = None,
        candidate_extractor: Optional[CandidateExtractor] = None,
        reports_dir: str = "reports",
        bootstrap_mode: bool = False,
        enable_event_logging: bool = True,
    ):
        self.locator_store = locator_store or LocatorStore()
        self.dom_capture = dom_capture or DOMCapture()
        self.ranker = ranker or HeuristicRanker(threshold=0.45)
        self.candidate_extractor = candidate_extractor or CandidateExtractor()
        self.reports_dir = ensure_dir(reports_dir)
        self.bootstrap_mode = bootstrap_mode

        # Continuous learning: log every healing decision with full features
        self._event_logger: Optional[Any] = None
        if enable_event_logging and _LOGGER_AVAILABLE:
            try:
                self._event_logger = HealEventLogger()
            except Exception:
                pass  # Graceful: logging is optional


    def find_element(
        self,
        driver: Any,
        test_case: str,
        page_name: str,
        element_name: str,
        locator_type: str,
        locator_value: str,
        timeout_seconds: int = 5,
        xpath_hint: str = "",
    ) -> Tuple[Optional[WebElement], Dict[str, Any]]:

        metadata_log: Dict[str, Any] = {
            "test_case": test_case,
            "page_name": page_name,
            "element_name": element_name,
            "original_locator": locator_value,
            "locator_type": locator_type,
            "requested_locator": f"{locator_type}={locator_value}",
            "healed": False,
            "healing_score": None,
            "healed_by": None,
            "status": "baseline_success",
            "error": "",
            "timestamp": timestamp(),
        }

        try:
            by = self._resolve_by(locator_type)

            element = driver.find_element(by, locator_value)

            self.locator_store.upsert_from_webelement(
                test_case=test_case,
                page_name=page_name,
                element_name=element_name,
                locator_type=locator_type,
                locator_value=locator_value,
                web_element=element,
                xpath_hint=xpath_hint,
            )

            return element, metadata_log

        except Exception as exc:
            metadata_log["status"] = "baseline_failed"
            metadata_log["error"] = truncate(str(exc), 300)


        # ---------- BOOTSTRAP MODE ----------
        if self.bootstrap_mode:
            raise NoSuchElementException(
                f"Bootstrap mode: locator failed for {test_case}/{element_name}. Healing disabled."
            )


        # ---------- HEALING START ----------
        if not self.locator_store.exists(test_case, element_name):
            raise NoSuchElementException(
                f"Locator failed for {test_case}/{element_name}, and no historical metadata exists for healing."
            )


        original = self.locator_store.load(test_case, element_name)

        dom_files = self.dom_capture.capture(driver, test_case, element_name)

        candidates = self.candidate_extractor.extract_candidates(
            driver.page_source,
            original.tag,
        )

        best = self.ranker.best_candidate(original, candidates)

        if not best:
            metadata_log["status"] = "heal_failed"

            # Log the no-candidate event
            self._log_heal_event(
                test_case=test_case, page_name=page_name,
                element_name=element_name, locator_value=locator_value,
                locator_type=locator_type, original=original,
                candidates=candidates, selected_idx=-1,
                selected_score=0.0, outcome="no_candidate",
                healed_locator="",
            )

            raise NoSuchElementException(
                f"Healing failed for {test_case}/{element_name}. "
                f"No candidate crossed threshold. DOM stored at {dom_files['html_file']}"
            )

        candidate: CandidateElement = best["candidate"]
        scores = best["scores"]

        healed_element = self._resolve_candidate_to_webelement(
            driver,
            candidate,
            original
        )

        if healed_element is None:
            metadata_log["status"] = "heal_failed_resolution"

            # Log the failed-resolution event
            selected_idx = candidates.index(candidate) if candidate in candidates else -1
            self._log_heal_event(
                test_case=test_case, page_name=page_name,
                element_name=element_name, locator_value=locator_value,
                locator_type=locator_type, original=original,
                candidates=candidates, selected_idx=selected_idx,
                selected_score=scores.get("total_score", 0.0),
                outcome="heal_failed_resolution",
                healed_locator=candidate.css_path_hint,
            )

            raise NoSuchElementException(
                f"Healing ranked a candidate but could not resolve it in the live DOM for {test_case}/{element_name}."
            )

        # Log the successful heal event
        selected_idx = candidates.index(candidate) if candidate in candidates else -1
        self._log_heal_event(
            test_case=test_case, page_name=page_name,
            element_name=element_name, locator_value=locator_value,
            locator_type=locator_type, original=original,
            candidates=candidates, selected_idx=selected_idx,
            selected_score=scores.get("total_score", 0.0),
            outcome="heal_success",
            healed_locator=candidate.css_path_hint,
            score_breakdown=scores,
        )

        metadata_log.update(
            {
                "healed": True,
                "healing_score": scores["total_score"],
                "healed_locator": candidate.css_path_hint,
                "healed_by": candidate.css_path_hint,
                "status": "heal_success",
                "healed_candidate": asdict(candidate),
                "score_breakdown": scores,
                "dom_file": dom_files["html_file"],
                "dom_meta_file": dom_files["meta_file"],
            }
        )

        return healed_element, metadata_log


    def _log_heal_event(
        self,
        test_case: str,
        page_name: str,
        element_name: str,
        locator_value: str,
        locator_type: str,
        original: Any,
        candidates: list,
        selected_idx: int,
        selected_score: float,
        outcome: str,
        healed_locator: str = "",
        score_breakdown: Optional[Dict] = None,
    ) -> None:
        """Log a complete healing event with features for all candidates."""
        if self._event_logger is None or not _LOGGER_AVAILABLE:
            return

        try:
            ts = timestamp()
            event_id = f"{test_case}__{element_name}__{ts}"

            # Build candidate records with full 13-feature vectors
            candidate_records = []
            for idx, cand in enumerate(candidates):
                # Compute features if available
                features = {}
                if _FEATURES_AVAILABLE:
                    try:
                        features = _compute_features(original, cand)
                    except Exception:
                        pass

                is_selected = (idx == selected_idx)
                label = 1 if (is_selected and outcome == "heal_success") else 0

                candidate_records.append(CandidateRecord(
                    candidate_idx=idx,
                    candidate_tag=getattr(cand, "tag", ""),
                    candidate_id=getattr(cand, "element_id", ""),
                    candidate_css_path=getattr(cand, "css_path_hint", ""),
                    features=features,
                    score=features.get("total_score", 0.0) if features else 0.0,
                    rank=idx + 1,
                    was_selected=is_selected,
                    label=label,
                ))

            # Determine ranker mode from type
            ranker_mode = type(self.ranker).__name__.lower()
            if "hybrid" in ranker_mode:
                ranker_mode = "hybrid"
            elif "ml" in ranker_mode:
                ranker_mode = "ml"
            elif "rule" in ranker_mode:
                ranker_mode = "rule_based"
            elif "random" in ranker_mode:
                ranker_mode = "random"
            else:
                ranker_mode = "heuristic"

            event = HealEvent(
                timestamp=ts,
                event_id=event_id,
                test_case=test_case,
                page_name=page_name,
                element_name=element_name,
                original_locator=locator_value,
                locator_type=locator_type,
                ui_version="",  # filled by caller if available
                original_tag=getattr(original, "tag", ""),
                n_candidates=len(candidates),
                ranker_mode=ranker_mode,
                selected_candidate_idx=selected_idx,
                selected_score=selected_score,
                outcome=outcome,
                healed_locator=healed_locator,
                score_breakdown=score_breakdown or {},
                candidates=candidate_records,
            )

            self._event_logger.log_event(event)

        except Exception:
            pass  # Never let logging break healing

    def flush_event_log(self) -> None:
        """Flush any buffered heal events to disk. Call at end of test run."""
        if self._event_logger is not None:
            self._event_logger.flush()

    def _resolve_by(self, locator_type: str):

        locator_type = locator_type.strip().lower()

        mapping = {
            "id": By.ID,
            "name": By.NAME,
            "xpath": By.XPATH,
            "css": By.CSS_SELECTOR,
            "css_selector": By.CSS_SELECTOR,
            "class": By.CLASS_NAME,
            "class_name": By.CLASS_NAME,
            "tag": By.TAG_NAME,
            "tag_name": By.TAG_NAME,
            "link_text": By.LINK_TEXT,
            "partial_link_text": By.PARTIAL_LINK_TEXT,
        }

        if locator_type not in mapping:
            raise ValueError(f"Unsupported locator type: {locator_type}")

        return mapping[locator_type]


    def _resolve_candidate_to_webelement(
        self,
        driver: Any,
        candidate: CandidateElement,
        original: Any,
    ) -> Optional[WebElement]:

        lookup_attempts = []

        if candidate.element_id:
            lookup_attempts.append((By.ID, candidate.element_id))

        if candidate.name:
            lookup_attempts.append((By.NAME, candidate.name))

        if getattr(original, "element_id", None):
            lookup_attempts.append((By.ID, original.element_id))

        if getattr(original, "name", None):
            lookup_attempts.append((By.NAME, original.name))

        if candidate.class_name and candidate.tag:
            first_class = candidate.class_name.split()[0]
            lookup_attempts.append((By.CSS_SELECTOR, f"{candidate.tag}.{first_class}"))

        seen = set()
        attempts = []

        for by, value in lookup_attempts:
            key = (by, value)
            if key not in seen:
                attempts.append((by, value))
                seen.add(key)

        for by, value in attempts:
            try:
                elements = driver.find_elements(by, value)

                if len(elements) >= 1:
                    return elements[0]

            except Exception:
                continue

        return None