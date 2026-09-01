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


from core.context_manager import ContextManager
from core.knowledge_router import KnowledgeRouter


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


def test_classify_image_shortcut_emits_intent():
    cm = _cm()
    router = KnowledgeRouter(None, None, context_manager=cm)
    verdict = router.classify("[image attached: x.png] interpret this")
    assert verdict == {"is_knowledge": False, "via": "image_shortcut"}
    intents = [e for e in cm.trace.events if e["t"] == "intent"]
    assert intents[0]["verdict"] == "TOOL" and intents[0]["via"] == "image_shortcut"


def test_classify_via_llm():
    class SimpleFake:
        def get_simple_completion(self, s, u, context_manager=None):
            return "KNOWLEDGE"
    cm = _cm()
    router = KnowledgeRouter(SimpleFake(), None, context_manager=cm)
    verdict = router.classify("How does frequency affect resolution")
    assert verdict == {"is_knowledge": True, "via": "llm"}


def test_classify_falls_back_to_keywords_and_records_it():
    class BrokenFake:
        def get_simple_completion(self, s, u, context_manager=None):
            raise RuntimeError("down")
    cm = _cm()
    router = KnowledgeRouter(BrokenFake(), None, context_manager=cm)
    verdict = router.classify("what is a ricker wavelet?")
    assert verdict == {"is_knowledge": True, "via": "keyword_fallback"}
    intents = [e for e in cm.trace.events if e["t"] == "intent"]
    assert intents[0]["via"] == "keyword_fallback"


def test_classify_tolerates_legacy_two_arg_fake():
    class LegacyFake:  # old signature without context_manager kwarg
        def get_simple_completion(self, s, u):
            return "TOOL"
    router = KnowledgeRouter(LegacyFake(), None, context_manager=_cm())
    assert router.classify("make a wedge model") == {"is_knowledge": False, "via": "llm"}


def test_handle_knowledge_question_emits_rag_scores():
    class FakeKB:
        def query_knowledge(self, q):
            return {"rag_type": "retrieve_and_generate", "generated_response": "answer",
                    "total_retrieved": 2,
                    "retrieved_documents": [{"score": 0.8123}, {"score": 0.5}]}
    cm = _cm()
    router = KnowledgeRouter(None, FakeKB(), context_manager=cm)
    out = router.handle_knowledge_question("what is tuning?")
    assert "answer" in out
    rag = [e for e in cm.trace.events if e["t"] == "rag"][0]
    assert rag["rag_type"] == "retrieve_and_generate"
    assert rag["retrieved"] == 2
    assert rag["scores"] == [0.8123, 0.5]
