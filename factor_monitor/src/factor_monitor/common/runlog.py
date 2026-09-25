"""Registro de avisos y estado de pasos, volcado a `run_meta.json`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("factor_monitor")


@dataclass
class RunLog:
    steps: dict[str, dict] = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)

    def warn(self, step: str, message: str, **extra) -> None:
        log.warning("[%s] %s", step, message)
        self.warnings.append({"step": step, "message": message, **extra})

    def step(self, name: str, status: str, **extra) -> None:
        self.steps[name] = {"status": status, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}

    def merge_into(self, path: Path) -> None:
        """Añade este registro a un `run_meta.json` existente (o lo crea)."""
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"steps": {}, "warnings": []}
        data["steps"].update(self.steps)
        data["warnings"].extend(self.warnings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
