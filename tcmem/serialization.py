from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value


def dump_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(to_primitive(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
