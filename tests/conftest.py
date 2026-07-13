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
    """Returns scripted completions; no network. Records each call's kwargs
    so tests can assert on the messages actually sent to the model."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get_completion(self, *a, **k):
        self.calls.append(k)
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory():
    return lambda responses: FakeLLMClient(responses)
