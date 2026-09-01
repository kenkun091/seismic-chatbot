"""Gradio upload path: staging into the sandbox + attaching to the session."""
import os

import pytest

from interfaces.gradio_interface import prepare_turn
from core.chatbot_tool_use import SeismicChatBotToolUse


def _bot():
    return SeismicChatBotToolUse(llm_client=object(), knowledge_base=object())


def test_no_image_passes_message_through(tmp_path):
    bot = _bot()
    assert prepare_turn("hello", None, bot, str(tmp_path), 10) == "hello"
    assert bot.context_manager.get_context("last_image") is None


def test_image_is_staged_attached_and_marked(outcrop_image, tmp_path):
    bot = _bot()
    base = str(tmp_path / "uploads")
    text = prepare_turn("what is this?", outcrop_image, bot, base, 10)
    staged = bot.context_manager.get_context("last_image")
    assert staged and staged.startswith(os.path.join(base, bot.session_id))
    assert text == f"[image attached: {os.path.basename(staged)}] what is this?"
    assert os.path.getsize(staged) == os.path.getsize(outcrop_image)


def test_image_with_empty_message_gets_default_request(outcrop_image, tmp_path):
    bot = _bot()
    text = prepare_turn("", outcrop_image, bot, str(tmp_path), 10)
    assert text.startswith("[image attached:") and "interpret" in text.lower()


def test_bad_upload_raises_value_error(tmp_path):
    bad = tmp_path / "x.gif"
    bad.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="extension"):
        prepare_turn("hi", str(bad), _bot(), str(tmp_path / "u"), 10)


def test_two_sessions_do_not_share_last_image(outcrop_image, tmp_path):
    base = _bot()
    a, b = base.new_session(), base.new_session()
    prepare_turn("x", outcrop_image, a, str(tmp_path), 10)
    assert a.context_manager.get_context("last_image")
    assert b.context_manager.get_context("last_image") is None
