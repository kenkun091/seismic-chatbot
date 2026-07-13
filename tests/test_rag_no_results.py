"""When RAG finds nothing above threshold, the assistant must not present an
unconstrained general-LLM answer as if it were grounded. It should label the
answer as not-from-the-knowledge-base and instruct the model not to fabricate
specific constants/citations."""
from core.chatbot_tool_use import SeismicChatBotToolUse


class _FakeLLM:
    last_system = None

    def get_simple_completion(self, system_prompt, user_prompt):
        _FakeLLM.last_system = system_prompt
        return "A general explanation."


def _bot_with_fake_llm():
    return SeismicChatBotToolUse(llm_client=_FakeLLM(), knowledge_base=object())


def test_no_results_answer_is_labelled_ungrounded():
    bot = _bot_with_fake_llm()
    out = bot._handle_no_rag_results("what is the Q factor of basalt at 2 km depth?")
    low = out.lower()
    assert "knowledge base" in low          # explicitly says it's not from the curated KB
    assert "general" in low


def test_no_results_prompt_discourages_fabrication():
    bot = _bot_with_fake_llm()
    bot._handle_no_rag_results("anything")
    sp = (_FakeLLM.last_system or "").lower()
    assert "do not" in sp or "avoid" in sp   # explicit anti-fabrication instruction
    assert "fabricat" in sp or "make up" in sp or "invent" in sp
