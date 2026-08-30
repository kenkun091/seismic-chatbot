from core.knowledge_router import KnowledgeRouter


class FakeSimpleLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def get_simple_completion(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class RaisingLLM:
    def get_simple_completion(self, *a, **k):
        raise RuntimeError("no network")


class FakeKB:
    def __init__(self, response):
        self._response = response

    def query_knowledge(self, q):
        return self._response

    def get_topic_response(self, topic, section):
        return f"canned:{topic}"


def test_image_attached_is_never_knowledge():
    router = KnowledgeRouter(FakeSimpleLLM("KNOWLEDGE"), FakeKB({}))
    assert router.is_knowledge_question("[image attached: x.png] interpret this") is False


def test_llm_classification_yes_and_no():
    assert KnowledgeRouter(FakeSimpleLLM("KNOWLEDGE"), FakeKB({})).is_knowledge_question("what is tuning") is True
    assert KnowledgeRouter(FakeSimpleLLM("TOOL"), FakeKB({})).is_knowledge_question("make a wavelet") is False


def test_keyword_fallback_when_llm_fails():
    router = KnowledgeRouter(RaisingLLM(), FakeKB({}))
    assert router.is_knowledge_question("What is a Ricker wavelet?") is True
    assert router.is_knowledge_question("make a 30 Hz ricker wavelet please") is False


def test_handle_knowledge_question_rag_hit():
    router = KnowledgeRouter(FakeSimpleLLM("unused"), FakeKB({
        "rag_type": "retrieve_and_generate",
        "generated_response": "Tuning is ...",
        "total_retrieved": 3,
    }))
    out = router.handle_knowledge_question("what is tuning")
    assert out.startswith("Tuning is ...")
    assert "3 relevant documents" in out


def test_handle_no_rag_results_appends_disclaimer():
    router = KnowledgeRouter(FakeSimpleLLM("General answer."), FakeKB({
        "rag_type": "no_results",
    }))
    out = router.handle_knowledge_question("what is obscure thing")
    assert "Not from the curated knowledge base" in out
