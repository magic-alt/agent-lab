from types import SimpleNamespace

from agent_lab.llm.openai_responses import OpenAIResponsesProvider


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        call = SimpleNamespace(
            type="function_call",
            call_id="call_123",
            name="calculate",
            arguments='{"expression":"2+2"}',
        )
        return SimpleNamespace(
            id="resp_1",
            output_text="",
            output=[call],
            usage={"input_tokens": 10},
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_provider_parses_function_call():
    client = FakeClient()
    provider = OpenAIResponsesProvider("test-model", client=client)
    step = provider.create(input_items="calculate", tools=[])
    assert step.response_id == "resp_1"
    assert step.tool_calls[0].name == "calculate"
    assert client.responses.kwargs["model"] == "test-model"
