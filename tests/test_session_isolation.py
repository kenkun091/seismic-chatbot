"""Per-session isolation: sessions must not share conversation state, but should
share the heavy, conversation-stateless components (LLM client, tools, KB)."""
from core.chatbot import SeismicChatBot
from core.chatbot_tool_use import SeismicChatBotToolUse


def _base_bot():
    # Inject cheap fakes for the heavy/credentialed deps so the test needs no
    # network or API keys; tool_manager defaults to the real (registry-only) one.
    return SeismicChatBotToolUse(llm_client=object(), knowledge_base=object())


def test_new_session_has_isolated_context():
    base = _base_bot()
    a = base.new_session()
    b = base.new_session()
    assert a.context_manager is not b.context_manager
    a.context_manager.set_context("last_ricker_wavelet", {"frequency": 30})
    assert b.context_manager.get_context("last_ricker_wavelet") is None


def test_new_session_shares_heavy_components():
    base = _base_bot()
    a = base.new_session()
    b = base.new_session()
    assert a.llm_client is base.llm_client
    assert a.tool_manager is base.tool_manager
    assert a.knowledge_base is base.knowledge_base
    assert a.tool_manager is b.tool_manager


def test_token_usage_is_per_session():
    base = _base_bot()
    a = base.new_session()
    b = base.new_session()
    a.context_manager.update_token_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert a.context_manager.get_token_usage()["total_tokens"] == 15
    assert b.context_manager.get_token_usage()["total_tokens"] == 0


def _legacy_base_bot():
    return SeismicChatBot(llm_client=object(), knowledge_base=object())


def test_legacy_new_session_has_isolated_context():
    base = _legacy_base_bot()
    a = base.new_session()
    b = base.new_session()
    assert a.context_manager is not b.context_manager
    a.context_manager.update_frequency(30)
    assert b.context_manager.get_last_frequency() is None


def test_legacy_new_session_shares_heavy_components():
    base = _legacy_base_bot()
    a = base.new_session()
    assert a.llm_client is base.llm_client
    assert a.tool_manager is base.tool_manager
    assert a.knowledge_base is base.knowledge_base
    assert a.input_parser is base.input_parser
