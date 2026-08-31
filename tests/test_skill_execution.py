import pytest
import yaml

from core.context_manager import ContextManager
from core.session_handle import SessionHandle
from core.skills import SkillRegistry, set_registry, validate_skill
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager
from core.tool_registry import REGISTRY_BY_NAME


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


def _session(llm=None):
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    tm = ToolManager()
    runner = ToolLoopRunner(llm, tm, cm)
    return SessionHandle(llm, tm, cm, runner), cm


def test_registry_declares_skill_tools_as_session_scoped():
    for name in ("run_skill", "save_skill", "list_skills"):
        assert REGISTRY_BY_NAME[name].session_scoped is True


def test_run_skill_replay_executes_chain_with_bound_params(registry):
    from tools.skill_tools import run_skill
    session, cm = _session()
    out = run_skill("ricker_wavelet", {"frequency": 45}, _session=session)
    assert out["mode"] == "replay"
    assert out["steps"] == [{"tool": "make_ricker", "ok": True}]
    assert out["extra_image_paths"] and out["extra_image_paths"][0].endswith(".png")
    assert cm.get_context("last_ricker_wavelet")["parameters"]["frequency"] == 45
    kinds = [e["t"] for e in cm.trace.events]
    assert "skill_run" in kinds and "auto_plot" in kinds
    import json
    sidecar = json.load(open(out["extra_image_paths"][0] + ".prov.json"))
    assert sidecar["skill"] == "ricker_wavelet"
    assert cm.get_context("_skill_depth") == 0


def test_run_skill_replay_stops_on_failed_step(registry):
    from tools.skill_tools import run_skill
    broken = dict(_RICKER, name="broken_ricker",
                  chain=[{"tool": "make_ricker", "args": {}},  # missing required frequency
                         {"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}])
    registry.save(broken)
    session, cm = _session()
    out = run_skill("broken_ricker", {}, _session=session)
    assert out["mode"] == "replay" and "error" in out
    assert out["steps"][0]["ok"] is False and len(out["steps"]) == 1
    assert any(e["t"] == "tool_call" and not e["ok"] for e in cm.trace.events)


def test_run_skill_guided_uses_executor(registry):
    from tools.skill_tools import run_skill
    llm = FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>built it</reply>", "tool_calls": None, "usage": None},
    ])
    session, cm = _session(llm)
    out = run_skill("ricker_wavelet", {"frequency": 30}, mode="guided", _session=session)
    assert out["mode"] == "guided" and out["summary"] == "built it"
    assert out["tools_used"] == ["make_ricker"]
    assert out["extra_image_paths"]


def test_run_skill_rejections(registry):
    from tools.skill_tools import run_skill
    session, _ = _session()
    with pytest.raises(ValueError):
        run_skill("nope", {}, _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {"bogus": 1}, _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, mode="weird", _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, _session=None)
    session.context_manager.set_context("_skill_depth", 1)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, _session=session)


def test_save_skill_captures_previous_turn(registry):
    from tools.skill_tools import list_skills, save_skill
    session, cm = _session()
    cm.set_context("last_turn_calls", [{"tool": "make_ricker", "args": {"frequency": 35}, "ok": True}])
    cm.set_context("last_turn_input", "make a 35 Hz ricker")
    out = save_skill("my_ricker", "My ricker", {"freq": 35}, _session=session)
    assert out["name"] == "my_ricker" and out["n_steps"] == 1
    saved = registry.get("my_ricker")
    assert saved.chain == [{"tool": "make_ricker", "args": {"frequency": "{{freq}}"}}]
    assert saved.procedure == "make a {{freq}} Hz ricker"
    assert [s["name"] for s in list_skills(_session=session)] == ["my_ricker", "ricker_wavelet"]
    with pytest.raises(ValueError):
        save_skill("my_ricker", "again", {"freq": 35}, _session=session)  # no overwrite


def test_save_skill_requires_a_prior_tool_turn(registry):
    from tools.skill_tools import save_skill
    session, cm = _session()
    with pytest.raises(ValueError):
        save_skill("x", "d", {"freq": 1}, _session=session)


def test_classic_prompt_and_orchestrator_prompt_mention_skills():
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
    prompt = SeismicChatBotToolUse._create_system_prompt(object.__new__(SeismicChatBotToolUse))
    assert "run_skill" in prompt and "save_skill" in prompt and "list_skills" in prompt
    assert "skill:" in ORCHESTRATOR_SYSTEM_PROMPT and "run_skill" in ORCHESTRATOR_SYSTEM_PROMPT
