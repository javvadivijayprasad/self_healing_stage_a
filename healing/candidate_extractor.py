from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from bs4 import BeautifulSoup, Tag


@dataclass
class CandidateElement:
    tag: str
    text: str
    element_id: str
    name: str
    class_name: str
    placeholder: str
    aria_label: str
    type_attr: str
    value_attr: str
    parent_tag: str
    sibling_count: int
    dom_depth: int
    css_path_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateExtractor:
    def __init__(self, max_candidates: int = 200) -> None:
        self.max_candidates = max_candidates

    def extract_candidates(self, page_source: str, preferred_tag: str = "") -> List[CandidateElement]:
        soup = BeautifulSoup(page_source, "lxml")
        tags = [preferred_tag] if preferred_tag else [
            "input", "button", "a", "select", "textarea", "label", "div", "span"
        ]

        results: List[CandidateElement] = []
        for tag_name in tags:
            for node in soup.find_all(tag_name):
                if not isinstance(node, Tag):
                    continue
                results.append(self._from_tag(node))
                if len(results) >= self.max_candidates:
                    return results

        if not results:
            for node in soup.find_all(True):
                if not isinstance(node, Tag):
                    continue
                results.append(self._from_tag(node))
                if len(results) >= self.max_candidates:
                    return results
        return results

    def _from_tag(self, node: Tag) -> CandidateElement:
        parent_tag = node.parent.name if isinstance(node.parent, Tag) else ""
        sibling_count = 0
        if isinstance(node.parent, Tag):
            sibling_count = len([child for child in node.parent.children if isinstance(child, Tag)])

        return CandidateElement(
            tag=(node.name or "").lower(),
            text=node.get_text(" ", strip=True),
            element_id=node.attrs.get("id", "") if node.attrs else "",
            name=node.attrs.get("name", "") if node.attrs else "",
            class_name=" ".join(node.attrs.get("class", [])) if node.attrs and node.attrs.get("class") else "",
            placeholder=node.attrs.get("placeholder", "") if node.attrs else "",
            aria_label=node.attrs.get("aria-label", "") if node.attrs else "",
            type_attr=node.attrs.get("type", "") if node.attrs else "",
            value_attr=node.attrs.get("value", "") if node.attrs else "",
            parent_tag=(parent_tag or "").lower(),
            sibling_count=sibling_count,
            dom_depth=self._dom_depth(node),
            css_path_hint=self._css_path_hint(node),
        )

    def _dom_depth(self, node: Tag) -> int:
        depth = 0
        current = node
        while current and isinstance(current.parent, Tag):
            depth += 1
            current = current.parent
        return depth

    def _css_path_hint(self, node: Tag) -> str:
        parts = []
        current = node
        while current and isinstance(current, Tag) and current.name != "[document]":
            part = current.name
            if current.get("id"):
                part += f"#{current.get('id')}"
                parts.append(part)
                break
            class_attr = current.get("class")
            if class_attr:
                part += "." + ".".join(class_attr[:2])
            parts.append(part)
            current = current.parent if isinstance(current.parent, Tag) else None
        return " > ".join(reversed(parts))
