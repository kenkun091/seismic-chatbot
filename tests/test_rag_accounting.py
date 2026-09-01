from core.context_manager import ContextManager
from core.knowledge_router import KnowledgeRouter
from knowledge.rag_system import RAGSystem


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


class AccountingFake:
    """Modern client: accepts context_manager and accounts tokens like the real one."""

    def __init__(self, reply="generated"):
        self.calls = []
        self.reply = reply

    def get_simple_completion(self, system_prompt, user_prompt, context_manager=None):
        self.calls.append(context_manager)
        if context_manager is not None:
            context_manager.update_token_usage(
                {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        return self.reply


_DOCS = [{"document": "Ricker wavelets are zero-phase.", "score": 0.8,
          "metadata": {"topic": "ricker", "subtopic": "overview"}}]


def test_generate_response_threads_context_manager():
    fake = AccountingFake()
    rag = object.__new__(RAGSystem)  # skip heavy __init__ (chroma vector db)
    rag.llm_client = fake
    cm = _cm()
    out = rag._generate_response("what is a ricker?", _DOCS, context_manager=cm)
    assert out == "generated"
    assert fake.calls == [cm]
    assert cm.get_token_usage()["total_tokens"] == 6


def test_generate_response_tolerates_legacy_two_arg_client():
    class Legacy:
        def get_simple_completion(self, s, u):
            return "legacy"

    rag = object.__new__(RAGSystem)
    rag.llm_client = Legacy()
    assert rag._generate_response("q", _DOCS, context_manager=_cm()) == "legacy"


def test_knowledge_base_injects_llm_client(monkeypatch):
    created = {}

    class FakeRAG:
        def __init__(self, llm_client=None):
            created["llm_client"] = llm_client

        def populate_knowledge_base(self, topics):
            pass

    monkeypatch.setattr("knowledge.knowledge_base.RAGSystem", FakeRAG)
    from knowledge.knowledge_base import KnowledgeBase
    sentinel = object()
    KnowledgeBase(llm_client=sentinel)
    assert created["llm_client"] is sentinel


def test_knowledge_base_query_passes_context_manager(monkeypatch):
    seen = {}

    class FakeRAG:
        def __init__(self, llm_client=None):
            pass

        def populate_knowledge_base(self, topics):
            pass

        def retrieve_and_generate(self, query, domain=None, context_manager=None):
            seen["cm"] = context_manager
            return {"rag_type": "no_results", "generated_response": "",
                    "retrieved_documents": [], "total_retrieved": 0}

    monkeypatch.setattr("knowledge.knowledge_base.RAGSystem", FakeRAG)
    from knowledge.knowledge_base import KnowledgeBase
    cm = _cm()
    KnowledgeBase().query_knowledge("q", context_manager=cm)
    assert seen["cm"] is cm


def test_router_passes_context_manager_with_legacy_fallback():
    cm = _cm()

    class ModernKB:
        def query_knowledge(self, q, domain=None, context_manager=None):
            self.cm = context_manager
            return {"rag_type": "retrieve_and_generate", "generated_response": "ans",
                    "total_retrieved": 1, "retrieved_documents": [{"score": 0.5}]}

    kb = ModernKB()
    router = KnowledgeRouter(None, kb, context_manager=cm)
    assert "ans" in router.handle_knowledge_question("what is tuning?")
    assert kb.cm is cm

    class LegacyKB:
        def query_knowledge(self, q):
            return {"rag_type": "retrieve_and_generate", "generated_response": "old",
                    "total_retrieved": 0, "retrieved_documents": []}

    router2 = KnowledgeRouter(None, LegacyKB(), context_manager=_cm())
    assert "old" in router2.handle_knowledge_question("what is tuning?")
