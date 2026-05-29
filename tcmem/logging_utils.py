from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .serialization import to_primitive


class ModuleLogStore:
    def __init__(self, base_dir: str | Path = "TCMem/logs", run_name: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.log("manifest", "run_started", base_dir=str(self.base_dir), run_name=self.run_name)

    def path_for(self, module: str) -> Path:
        return self.run_dir / f"{module}.jsonl"

    def log(self, module: str, event: str, **payload: Any) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "module": module,
            "event": event,
            "payload": to_primitive(payload),
        }
        target = self.path_for(module)
        with self._lock:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
