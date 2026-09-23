from __future__ import annotations

import json
from typing import Any

from agent_lab.tools.registry import ToolRegistry
from agent_lab.tracing import TraceWriter


def run_tool_loop(
    *,
    provider: Any,
    prompt: str,
    registry: ToolRegistry,
    instructions: str = (
        "Use tools when they are needed. Never invent a tool result. "
        "Return a concise final answer after tool execution."
    ),
    max_steps: int = 6,
    trace: TraceWriter | None = None,
) -> str:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    input_items: list[Any] = [{"role": "user", "content": prompt}]
    for step_index in range(max_steps):
        step = provider.create(
            input_items=input_items,
            instructions=instructions,
            tools=registry.schemas(),
        )
        input_items.extend(step.raw_output_items)
        if not step.tool_calls:
            if trace:
                trace.emit("agent.finished", step=step_index, response_id=step.response_id)
            return step.output_text

        for call in step.tool_calls:
            execution = registry.execute(
                call_id=call.call_id,
                name=call.name,
                arguments_json=call.arguments_json,
            )
            if trace:
                trace.emit("tool.result", execution=execution)
            output = execution.output
            if not execution.ok:
                output = json.dumps(
                    {"ok": False, "error": execution.output}, ensure_ascii=False
                )
            input_items.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )

    if trace:
        trace.emit("agent.terminated", reason="max_steps", max_steps=max_steps)
    raise RuntimeError(f"tool loop exceeded max_steps={max_steps}")
