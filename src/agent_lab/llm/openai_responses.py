from __future__ import annotations

import time
from typing import Any

from agent_lab.tracing import TraceWriter
from agent_lab.types import ModelStep, ToolCall


class OpenAIResponsesProvider:
    """Thin adapter around OpenAI's Responses API.

    `client` is injectable so all protocol behavior can be tested offline.
    """

    def __init__(self, model: str, *, client: Any | None = None, trace: TraceWriter | None = None):
        if not model:
            raise ValueError("model must be non-empty")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.model = model
        self.client = client
        self.trace = trace

    def create(
        self,
        *,
        input_items: str | list[dict[str, Any]] | list[Any],
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelStep:
        kwargs: dict[str, Any] = {"model": self.model, "input": input_items}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools

        started = time.perf_counter()
        if self.trace:
            self.trace.emit("model.request", model=self.model, has_tools=bool(tools))
        response = self.client.responses.create(**kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        step = self._parse(response)
        if self.trace:
            self.trace.emit(
                "model.response",
                model=self.model,
                response_id=step.response_id,
                output_text=step.output_text,
                tool_calls=step.tool_calls,
                usage=step.usage,
                elapsed_ms=round(elapsed_ms, 3),
            )
        return step

    @staticmethod
    def _parse(response: Any) -> ModelStep:
        output_items = tuple(getattr(response, "output", ()) or ())
        calls: list[ToolCall] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                calls.append(
                    ToolCall(
                        call_id=str(getattr(item, "call_id")),
                        name=str(getattr(item, "name")),
                        arguments_json=str(getattr(item, "arguments", "{}")),
                    )
                )
        usage_obj = getattr(response, "usage", None)
        usage: dict[str, Any] = {}
        if usage_obj is not None:
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = dict(usage_obj)
        return ModelStep(
            response_id=str(getattr(response, "id", "")),
            output_text=str(getattr(response, "output_text", "") or ""),
            tool_calls=tuple(calls),
            raw_output_items=output_items,
            usage=usage,
        )
