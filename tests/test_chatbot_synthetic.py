"""Chatbot wiring for the N-layer synthetic: context, auto-plot chain, prompt."""
import os

import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse
from tools.synthetic_tools import create_synthetic_seismogram


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


@pytest.fixture
def synthetic_result():
    return create_synthetic_seismogram(
        [50.0, 50.0], [3000.0, 2500.0, 3200.0], [2.4, 2.2, 2.5])


def test_update_context_stores_last_synthetic(bot, synthetic_result):
    bot._update_context("synthetic_seismogram", {"vp": [3000.0, 2500.0, 3200.0]},
                        synthetic_result)
    stored = bot.context_manager.get_context("last_synthetic")
    assert stored is not None
    assert stored["parameters"]["n_layers"] == 3
    assert stored["input_params"] == {"vp": [3000.0, 2500.0, 3200.0]}


def test_auto_chain_plots_from_context(bot, synthetic_result):
    bot._update_context("synthetic_seismogram", {}, synthetic_result)
    chained = bot._handle_automatic_chaining("synthetic_seismogram", {},
                                             synthetic_result)
    assert chained is not None and chained["image_path"].endswith(".png")
    assert os.path.getsize(chained["image_path"]) > 0
    os.remove(chained["image_path"])


def test_auto_chain_without_context_returns_none(bot, synthetic_result):
    chained = bot._handle_automatic_chaining("synthetic_seismogram", {},
                                             synthetic_result)
    assert chained is None


def test_system_prompt_lists_synthetic(bot):
    assert "- synthetic_seismogram:" in bot._create_system_prompt()
