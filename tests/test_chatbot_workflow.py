import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse


@pytest.fixture
def bot(fake_llm_factory):
    # fake_llm_factory(responses) -> a no-network FakeLLMClient (tests/conftest.py).
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


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


def test_system_prompt_lists_saturation(bot):
    prompt = bot._create_system_prompt()
    assert "- rock_properties_saturation:" in prompt
    assert "- saturation_sweep:" in prompt


def test_system_prompt_lists_run_sweep(bot):
    prompt = bot._create_system_prompt()
    assert "- run_sweep:" in prompt


def test_harvest_images_from_plain_path(bot):
    collected = []
    bot._harvest_images("/tmp/a.png", collected)
    assert collected == ["/tmp/a.png"]


def test_harvest_images_from_dict(bot):
    collected = []
    bot._harvest_images({"avo_class": "III", "image_path": "/tmp/x.png"}, collected)
    assert collected == ["/tmp/x.png"]


def test_harvest_images_dedupes_preserving_order(bot):
    collected = ["/tmp/a.png"]
    bot._harvest_images({"image_path": "/tmp/b.png"}, collected)
    bot._harvest_images("/tmp/a.png", collected)
    bot._harvest_images({"image_path": "/tmp/b.png"}, collected)
    assert collected == ["/tmp/a.png", "/tmp/b.png"]


def test_harvest_images_ignores_non_images(bot):
    collected = []
    bot._harvest_images({"avo_class": "III"}, collected)
    bot._harvest_images({"image_path": 123}, collected)
    bot._harvest_images("not-a-path", collected)
    bot._harvest_images(("tuple", "result"), collected)
    bot._harvest_images(None, collected)
    assert collected == []
