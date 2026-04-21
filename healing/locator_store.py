from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from healing.utils import ensure_dir, read_json, write_json


@dataclass
class LocatorMetadata:
    test_case: str
    page_name: str
    element_name: str
    locator_type: str
    locator_value: str
    tag: str = ""
    text: str = ""
    element_id: str = ""
    name: str = ""
    class_name: str = ""
    placeholder: str = ""
    aria_label: str = ""
    xpath_hint: str = ""
    parent_tag: str = ""
    dom_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocatorStore:
    def __init__(self, root_dir: str = "data/processed/locator_store") -> None:
        self.root_dir = ensure_dir(root_dir)

    def _path(self, test_case: str, element_name: str) -> Path:
        return self.root_dir / f"{test_case}__{element_name}.json"

    def save(self, metadata: LocatorMetadata) -> Path:
        path = self._path(metadata.test_case, metadata.element_name)
        write_json(path, metadata.to_dict())
        return path

    def load(self, test_case: str, element_name: str) -> LocatorMetadata:
        path = self._path(test_case, element_name)
        raw = read_json(path)
        return LocatorMetadata(**raw)

    def exists(self, test_case: str, element_name: str) -> bool:
        return self._path(test_case, element_name).exists()

    def upsert_from_webelement(
        self,
        *,
        test_case: str,
        page_name: str,
        element_name: str,
        locator_type: str,
        locator_value: str,
        web_element: Any,
        xpath_hint: str = "",
    ) -> LocatorMetadata:
        parent_tag = ""
        dom_depth = 0
        try:
            parent = web_element.find_element("xpath", "..")
            parent_tag = (parent.tag_name or "").lower()
        except Exception:
            parent_tag = ""
        try:
            dom_depth = int(
                web_element.parent.execute_script(
                    """
                    function getDepth(el) {
                        let depth = 0;
                        while (el && el.parentElement) {
                            depth++;
                            el = el.parentElement;
                        }
                        return depth;
                    }
                    return getDepth(arguments[0]);
                    """,
                    web_element,
                )
            )
        except Exception:
            dom_depth = 0

        metadata = LocatorMetadata(
            test_case=test_case,
            page_name=page_name,
            element_name=element_name,
            locator_type=locator_type,
            locator_value=locator_value,
            tag=(web_element.tag_name or "").lower(),
            text=(web_element.text or "").strip(),
            element_id=web_element.get_attribute("id") or "",
            name=web_element.get_attribute("name") or "",
            class_name=web_element.get_attribute("class") or "",
            placeholder=web_element.get_attribute("placeholder") or "",
            aria_label=web_element.get_attribute("aria-label") or "",
            xpath_hint=xpath_hint,
            parent_tag=parent_tag,
            dom_depth=dom_depth,
        )
        self.save(metadata)
        return metadata
