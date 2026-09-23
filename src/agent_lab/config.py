from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str
    trace_dir: Path

    @classmethod
    def from_env(cls, *, require_model: bool = False) -> "Settings":
        model = os.getenv("AGENT_LAB_MODEL", "").strip()
        if require_model and not model:
            raise RuntimeError(
                "AGENT_LAB_MODEL is not set. Choose a model available to your OpenAI API project."
            )
        trace_dir = Path(os.getenv("AGENT_LAB_TRACE_DIR", "runs"))
        return cls(model=model, trace_dir=trace_dir)
