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


def test_update_context_caches_workflow_result(bot):
    result = {"avo_class": "III", "image_path": "/tmp/x.png"}
    bot._update_context("petro_to_avo", {"phit_sand": 0.25}, result)
    assert bot.context_manager.get_context("last_workflow_result") == result


def test_system_prompt_lists_petro_to_avo(bot):
    prompt = bot._create_system_prompt()
    assert "petro_to_avo" in prompt


def test_system_prompt_lists_fluid_scenario(bot):
    prompt = bot._create_system_prompt()
    assert "fluid_scenario" in prompt


def test_system_prompt_lists_tuning(bot):
    # Match the bullet prefix specifically: the word "tuning" may already appear
    # inside other wedge-tool descriptions, but "- tuning:" is the new bullet.
    prompt = bot._create_system_prompt()
    assert "- tuning:" in prompt


def test_system_prompt_lists_eei_optimal_chi(bot):
    prompt = bot._create_system_prompt()
    assert "- eei_optimal_chi:" in prompt
    assert "- eei_optimal_chi_petro:" in prompt
