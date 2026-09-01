import json

import pytest
import yaml

from core.context_manager import ContextManager
from core.skills import CONTEXT_PARAMS, SkillRegistry, set_registry, validate_skill
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunc(name, arguments)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def get_completion(self, *a, **k):
        return self._responses.pop(0)


_RICKER = {
    "name": "ricker_wavelet", "description": "Build a Ricker wavelet.",
    "parameters": {"frequency": {"type": "number", "description": "Hz", "default": 30}},
    "tools": ["make_ricker"], "procedure": "Create a {{frequency}} Hz Ricker wavelet.",
    "chain": [{"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}],
}


@pytest.fixture
def registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ricker_wavelet.yaml").write_text(yaml.safe_dump(_RICKER, sort_keys=False))
    reg = SkillRegistry(repo_dir=str(repo), runtime_dir=str(tmp_path / "rt"))
    set_registry(reg)
    yield reg
    set_registry(None)


def test_session_scoped_tools_are_not_recorded(registry):
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    runner.execute_call("list_skills", {}, [])
    runner.execute_call("run_skill", {"name": "ricker_wavelet", "params": {"frequency": 35}}, [])
    tools = [c["tool"] for c in cm.get_context("current_turn_calls")]
    assert tools == ["make_ricker"]  # the inner step is recorded; run_skill itself is not


def test_validate_rejects_session_scoped_tools():
    with pytest.raises(ValueError) as exc:
        validate_skill(dict(_RICKER, tools=["make_ricker", "run_skill"]))
    assert "session-scoped" in str(exc.value)


def test_orchestrator_prompt_hides_internal_keys():
    from core.orchestrator import SeismicOrchestrator
    from core.tool_manager import ToolManager as TM
    orch = SeismicOrchestrator(llm_client=object(), tool_manager=TM(),
                               knowledge_base=object(), tool_index=object())
    orch.context_manager.trace.persist_dir = ""
    orch.context_manager.begin_turn_recording("hello")
    orch.context_manager.set_context("_skill_depth", 0)
    assert "fresh conversation" in orch._system_prompt()
    orch.context_manager.set_context("last_ricker_wavelet", {"x": 1})
    prompt = orch._system_prompt()
    assert "last_ricker_wavelet" in prompt and "current_turn" not in prompt


def test_guided_mode_sidecar_carries_skill(registry):
    from core.session_handle import SessionHandle
    from tools.skill_tools import run_skill
    llm = FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>built</reply>", "tool_calls": None, "usage": None},
    ])
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    tm = ToolManager()
    session = SessionHandle(llm, tm, cm, ToolLoopRunner(llm, tm, cm))
    out = run_skill("ricker_wavelet", {"frequency": 30}, mode="guided", _session=session)
    sidecar = json.load(open(out["extra_image_paths"][0] + ".prov.json"))
    assert sidecar["skill"] == "ricker_wavelet"


def test_context_params_mirror_loop_context_inputs():
    assert set(CONTEXT_PARAMS) == {p for _, p, _ in ToolLoopRunner._CONTEXT_INPUTS}


def test_classic_round_trip_save_then_run(registry):
    from core.chatbot_tool_use import SeismicChatBotToolUse
    llm = FakeLLM([
        # turn 1: make a wavelet
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')], "usage": None},
        {"content": "<reply>made</reply>", "tool_calls": None, "usage": None},
        # turn 2: save the previous turn as a skill
        {"content": "", "tool_calls": [FakeToolCall(
            "save_skill", '{"name": "rt", "description": "round trip", "parameters": {"freq": 30}}')],
         "usage": None},
        {"content": "<reply>saved</reply>", "tool_calls": None, "usage": None},
        # turn 3: run it at a new frequency
        {"content": "", "tool_calls": [FakeToolCall(
            "run_skill", '{"name": "rt", "params": {"freq": 45}}')], "usage": None},
        {"content": "<reply>ran</reply>", "tool_calls": None, "usage": None},
    ])
    bot = SeismicChatBotToolUse(llm_client=llm, tool_manager=ToolManager(), knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    bot.process_single_input("make a 30 Hz ricker wavelet")
    bot.process_single_input("save that as a skill named rt")
    assert registry.get("rt").chain == [{"tool": "make_ricker", "args": {"frequency": "{{freq}}"}}]
    out = bot.process_single_input("run skill rt at 45 Hz")
    assert out["images"] and out["images"][0].endswith(".png")
    assert bot.context_manager.get_context("last_ricker_wavelet")["parameters"]["frequency"] == 45
