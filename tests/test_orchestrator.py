import json
import pytest
from core.orchestrator import SeismicOrchestrator, META_TOOL_NAMES, MAX_ORCH_ROUNDS
from core.tool_manager import ToolManager
from core.context_manager import ContextManager
from core.tool_index import ToolCard


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


class FakeToolIndex:
    def __init__(self, cards):
        self._cards = cards
        self.queries = []

    def search(self, task_description, top_k=5, threshold=0.2):
        self.queries.append(task_description)
        return self._cards


class FakeKB:
    def query_knowledge(self, q):
        return {"rag_type": "retrieve_and_generate", "generated_response": "kb", "total_retrieved": 1}

    def get_topic_response(self, topic, section):
        return "canned"


RICKER_CARD = ToolCard(name="make_ricker",
                       card="make_ricker: Creates a Ricker wavelet. Params: frequency (number, required)",
                       required=("frequency",), score=0.8)


def make_orchestrator(fake_llm_factory, responses, cards=(RICKER_CARD,)):
    llm = fake_llm_factory(responses)
    orch = SeismicOrchestrator(llm_client=llm, tool_manager=ToolManager(),
                               knowledge_base=FakeKB(), tool_index=FakeToolIndex(list(cards)))
    return orch, llm


def final(text):
    return {"content": text, "tool_calls": None, "stop_reason": "stop", "usage": None}


def meta(name, args, call_id="c1"):
    return {"content": "", "stop_reason": "tool_calls", "usage": None,
            "tool_calls": [FakeToolCall(name, json.dumps(args), call_id)]}


def test_meta_schemas_are_the_only_tools_sent(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>hi</reply>")])
    orch.process_single_input("make me a wavelet")
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == set(META_TOOL_NAMES) == {"discover_tools", "run_task"}


def test_discover_then_delegate_then_compose(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("discover_tools", {"task_description": "create ricker wavelet"}),
        meta("run_task", {"brief": "Create a 30 Hz Ricker wavelet.",
                          "tool_names": ["make_ricker"]}, "c2"),
        # executor's two calls:
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        final("<reply>Executor: built it.</reply>"),
        # orchestrator composes:
        final("<reply>Your 30 Hz Ricker wavelet is ready.</reply>"),
    ])
    out = orch.process_single_input("create a 30 Hz ricker wavelet")
    assert out["reply"] == "Your 30 Hz Ricker wavelet is ready."
    assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
    # FakeLLMClient records a REFERENCE to the orchestrator's messages list,
    # so calls[0]["messages"] shows the final state; filter by role, don't index.
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "make_ricker:" in tool_msgs[0]["content"]        # discovery result card
    assert "Executor: built it." in tool_msgs[1]["content"]  # run_task summary
    assert ".png" not in tool_msgs[1]["content"]             # image paths stay out-of-band


def test_unknown_tool_name_reported_no_executor(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("run_task", {"brief": "x", "tool_names": ["bogus_tool"]}),
        final("<reply>Cannot do that.</reply>"),
    ])
    out = orch.process_single_input("do something odd")
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "bogus_tool" in tool_msgs[0]["content"]
    assert out["reply"] == "Cannot do that."


def test_empty_discovery_gets_informative_message(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("discover_tools", {"task_description": "cook pasta"}),
        final("<reply>I have no tools for that.</reply>"),
    ], cards=())
    orch.process_single_input("cook pasta")
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "No tools matched" in tool_msgs[0]["content"]


def test_round_budget_forces_tool_free_completion(fake_llm_factory):
    responses = [meta("discover_tools", {"task_description": "x"}, f"c{i}")
                 for i in range(MAX_ORCH_ROUNDS)]
    responses.append(final("<reply>Ran out of rounds.</reply>"))
    orch, llm = make_orchestrator(fake_llm_factory, responses)
    out = orch.process_single_input("loop forever")
    assert out["reply"] == "Ran out of rounds."
    assert llm.calls[-1]["tools"] is None


def test_knowledge_question_routes_to_rag_not_meta_loop(fake_llm_factory):
    # get_simple_completion is absent on FakeLLMClient -> keyword fallback; '?' => knowledge
    orch, llm = make_orchestrator(fake_llm_factory, [])
    out = orch.process_single_input("What is seismic tuning?")
    assert out["reply"].startswith("kb")
    assert llm.calls == []  # no meta-loop LLM calls


def test_context_keys_line_in_system_prompt(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>ok</reply>")])
    orch.context_manager.set_context("last_wedge_model", {"x": 1})
    orch.process_single_input("tweak the wedge")
    assert "last_wedge_model" in llm.calls[0]["system_prompt"]


def test_attach_image_and_image_message_route_to_tools(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>photo noted</reply>")])
    orch.attach_image("/sandbox/s1/photo.png")
    out = orch.process_single_input("[image attached: photo.png] interpret this")
    assert out["reply"] == "photo noted"
    assert orch.context_manager.get_context("last_image") == "/sandbox/s1/photo.png"


def test_non_dict_meta_args_reported_not_raised(fake_llm_factory):
    # A JSON-valid-but-non-object arguments string (e.g. "[]") must not raise
    # AttributeError out of _dispatch_meta -- it should become a recoverable
    # tool message so the turn survives.
    non_dict_call = {"content": "", "stop_reason": "tool_calls", "usage": None,
                     "tool_calls": [FakeToolCall("run_task", "[]")]}
    orch, llm = make_orchestrator(fake_llm_factory, [
        non_dict_call,
        final("<reply>Recovered.</reply>"),
    ])
    out = orch.process_single_input("do something odd")
    assert out["reply"] == "Recovered."
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "Invalid arguments" in tool_msgs[0]["content"]


def test_run_task_rejects_empty_tool_names_before_spawning_executor(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("run_task", {"brief": "x", "tool_names": []}),
        final("<reply>Cannot do that.</reply>"),
    ])
    out = orch.process_single_input("do something odd")
    # Only the two orchestrator-loop responses should have been consumed;
    # no executor LLM call should have popped a scripted response.
    assert llm.calls == llm.calls  # sanity: calls recorded
    assert len(llm.calls) == 2
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "tool_names is empty" in tool_msgs[0]["content"]
    assert out["reply"] == "Cannot do that."


def test_run_task_rejects_non_list_tool_names(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("run_task", {"brief": "x", "tool_names": "make_ricker"}),
        final("<reply>Cannot do that.</reply>"),
    ])
    out = orch.process_single_input("do something odd")
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "m, a, k, e" not in tool_msgs[0]["content"]
    assert "tool_names is empty" in tool_msgs[0]["content"] or \
        "tool_names must be a non-empty list" in tool_msgs[0]["content"]
    assert out["reply"] == "Cannot do that."
