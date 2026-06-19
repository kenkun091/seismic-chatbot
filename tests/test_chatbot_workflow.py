import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse


@pytest.fixture
def bot(fake_llm_factory):
    # fake_llm_factory(responses) -> a no-network FakeLLMClient (tests/conftest.py).
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


def test_workflow_image_output_from_dict(bot):
    out = bot._workflow_image_output({"avo_class": "III", "image_path": "/tmp/x.png"})
    assert out == {"image_path": "/tmp/x.png"}


def test_workflow_image_output_none_when_no_png(bot):
    assert bot._workflow_image_output({"avo_class": "III"}) is None
    assert bot._workflow_image_output({"image_path": 123}) is None
    assert bot._workflow_image_output("not-a-dict") is None
