import pytest

from core.skills import SkillRegistry, set_registry
from interfaces.gradio_interface import (parse_parameter_lines, save_skill_from_ui,
                                         skills_markdown)


@pytest.fixture
def registry(tmp_path):
    reg = SkillRegistry(repo_dir=str(tmp_path / "none"), runtime_dir=str(tmp_path / "rt"))
    set_registry(reg)
    yield reg
    set_registry(None)


def test_parse_parameter_lines():
    assert parse_parameter_lines("freq=30\nphit = 0.25\nname=sand\n\n") == \
        {"freq": 30, "phit": 0.25, "name": "sand"}
    with pytest.raises(ValueError):
        parse_parameter_lines("no equals sign")


def test_save_skill_from_ui_uses_last_completed_turn(registry):
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.tool_manager import ToolManager
    bot = SeismicChatBotToolUse(llm_client=object(), tool_manager=ToolManager(),
                                knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    bot.context_manager.set_context("current_turn_calls",
                                    [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}])
    bot.context_manager.set_context("current_turn_input", "make a 30 Hz ricker")
    status = save_skill_from_ui(bot, "ui_ricker", "From the UI", "freq=30")
    assert "Saved skill" in status and "ui_ricker" in status
    assert registry.get("ui_ricker").chain[0]["args"] == {"frequency": "{{freq}}"}
    assert "ui_ricker" in skills_markdown()


def test_save_skill_from_ui_reports_errors(registry):
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.tool_manager import ToolManager
    bot = SeismicChatBotToolUse(llm_client=object(), tool_manager=ToolManager(),
                                knowledge_base=object())
    assert "no tools" in save_skill_from_ui(bot, "x", "d", "freq=1").lower()
    assert save_skill_from_ui(None, "x", "d", "freq=1").startswith("⚠️")
