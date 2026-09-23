from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class TraceWriter:
    """Very small local JSONL trace writer for observable engineering events."""

    def __init__(self, root: Path, *, run_id: str | None = None) -> None:
        self.run_id = run_id or f"run_{uuid4().hex}"
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{self.run_id}.jsonl"

    def emit(self, event: str, **data: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
