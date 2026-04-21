from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from healing.utils import ensure_dir, safe_filename, timestamp, write_json


class DOMCapture:
    def __init__(self, raw_dom_dir: str = "data/raw_dom") -> None:
        self.raw_dom_dir = ensure_dir(raw_dom_dir)

    def capture(self, driver: Any, test_case: str, step_name: str) -> Dict[str, str]:
        page_source = driver.page_source
        page_title = driver.title
        current_url = driver.current_url
        ts = timestamp()
        base_name = f"{safe_filename(test_case)}__{safe_filename(step_name)}__{ts}"

        html_path = self.raw_dom_dir / f"{base_name}.html"
        meta_path = self.raw_dom_dir / f"{base_name}.json"

        html_path.write_text(page_source, encoding="utf-8")
        write_json(
            meta_path,
            {
                "test_case": test_case,
                "step_name": step_name,
                "title": page_title,
                "url": current_url,
                "html_file": str(html_path),
                "timestamp": ts,
            },
        )
        return {
            "html_file": str(html_path),
            "meta_file": str(meta_path),
            "title": page_title,
            "url": current_url,
        }
