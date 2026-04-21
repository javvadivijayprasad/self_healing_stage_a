from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def safe_filename(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return value[:200] if value else "file"


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def truncate(value: Optional[str], limit: int = 120) -> str:
    if value is None:
        return ""
    value = str(value)
    return value if len(value) <= limit else value[: limit - 3] + "..."
