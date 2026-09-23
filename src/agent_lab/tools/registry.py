from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent_lab.types import ToolExecution


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Any]

    def openai_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": schema,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, *, call_id: str, name: str, arguments_json: str) -> ToolExecution:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecution(call_id=call_id, name=name, ok=False, output="unknown_tool")
        try:
            raw = json.loads(arguments_json)
            parsed = tool.input_model.model_validate(raw)
            kwargs = parsed.model_dump()
            result = tool.handler(**kwargs)
            if inspect.isawaitable(result):
                raise TypeError("async handlers are not supported by the Phase-04 sync registry")
            output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            return ToolExecution(call_id=call_id, name=name, ok=True, output=output)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError, OSError) as exc:
            return ToolExecution(
                call_id=call_id,
                name=name,
                ok=False,
                output=json.dumps(
                    {"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
                ),
            )
