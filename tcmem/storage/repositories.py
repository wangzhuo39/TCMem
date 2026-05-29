from __future__ import annotations

from pathlib import Path
from typing import Any

from ..serialization import dump_json, load_json


class FileSystemMemoryRepository:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.state_path = self.storage_path / "memory_state.json"

    def save_state(self, state: dict[str, Any]) -> None:
        dump_json(self.state_path, state)

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return load_json(self.state_path)

    def exists(self) -> bool:
        return self.state_path.exists()
