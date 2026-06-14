import pytest


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunc(name, arguments)


class FakeLLMClient:
    """Returns scripted completions; no network."""
    def __init__(self, responses):
        self._responses = list(responses)

    def get_completion(self, *a, **k):
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory():
    return lambda responses: FakeLLMClient(responses)
