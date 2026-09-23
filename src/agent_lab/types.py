from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ModelStep:
    response_id: str
    output_text: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw_output_items: tuple[Any, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecution:
    call_id: str
    name: str
    ok: bool
    output: str
