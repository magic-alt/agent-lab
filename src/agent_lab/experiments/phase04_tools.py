from __future__ import annotations

from pathlib import Path

from agent_lab.config import Settings
from agent_lab.llm import OpenAIResponsesProvider
from agent_lab.tool_loop import run_tool_loop
from agent_lab.tools import build_default_registry
from agent_lab.tracing import TraceWriter


def run(prompt: str, *, file_root: Path | None = None) -> str:
    settings = Settings.from_env(require_model=True)
    trace = TraceWriter(settings.trace_dir)
    provider = OpenAIResponsesProvider(settings.model, trace=trace)
    registry = build_default_registry(file_root=file_root)
    answer = run_tool_loop(
        provider=provider,
        prompt=prompt,
        registry=registry,
        max_steps=6,
        trace=trace,
    )
    print(f"trace: {trace.path}")
    return answer
