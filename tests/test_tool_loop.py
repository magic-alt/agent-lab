from types import SimpleNamespace

from agent_lab.tool_loop import run_tool_loop
from agent_lab.tools import build_default_registry
from agent_lab.types import ModelStep, ToolCall


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.seen_inputs = []

    def create(self, *, input_items, instructions=None, tools=None):
        self.calls += 1
        self.seen_inputs.append(list(input_items))
        if self.calls == 1:
            raw = SimpleNamespace(
                type="function_call",
                call_id="c1",
                name="calculate",
                arguments='{"expression":"6*7"}',
            )
            return ModelStep(
                response_id="r1",
                output_text="",
                tool_calls=(ToolCall("c1", "calculate", '{"expression":"6*7"}'),),
                raw_output_items=(raw,),
            )
        return ModelStep(response_id="r2", output_text="42", raw_output_items=())


def test_manual_tool_loop_round_trip():
    provider = FakeProvider()
    answer = run_tool_loop(
        provider=provider,
        prompt="What is 6*7?",
        registry=build_default_registry(),
        max_steps=3,
    )
    assert answer == "42"
    assert provider.calls == 2
    second = provider.seen_inputs[1]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second
    )
