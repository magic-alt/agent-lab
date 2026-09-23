from __future__ import annotations

from agent_lab.config import Settings
from agent_lab.llm import OpenAIResponsesProvider
from agent_lab.tracing import TraceWriter


def run(prompt: str) -> str:
    settings = Settings.from_env(require_model=True)
    trace = TraceWriter(settings.trace_dir)
    provider = OpenAIResponsesProvider(settings.model, trace=trace)
    step = provider.create(
        input_items=prompt,
        instructions="Answer accurately and concisely. State uncertainty when material.",
    )
    print(f"trace: {trace.path}")
    return step.output_text
