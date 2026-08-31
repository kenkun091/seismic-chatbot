from core.context_manager import ContextManager
from core.session_handle import SessionHandle
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager
from core.tool_registry import ToolSpec


def _scoped_spec(fn):
    return ToolSpec(name="probe_session", fn=fn, description="probe", params={},
                    required=[], session_scoped=True)


def test_execute_tool_passes_session_only_to_scoped_tools():
    seen = {}

    def probe(_session=None):
        seen["session"] = _session
        return "ok"

    tm = ToolManager()
    tm.specs = dict(tm.specs, probe_session=_scoped_spec(probe))
    tm.tools["probe_session"] = probe
    handle = object()
    assert tm.execute_tool("probe_session", {"_session": handle}) == "ok"
    assert seen["session"] is handle
    # a non-scoped tool never receives _session even if present
    tm.execute_tool("make_ricker", {"frequency": 30, "_session": handle})


def test_loop_injects_session_handle_and_keeps_it_out_of_events_and_recording():
    seen = {}

    def probe(_session=None):
        seen["session"] = _session
        return {"value": 1}

    tm = ToolManager()
    tm.specs = dict(tm.specs, probe_session=_scoped_spec(probe))
    tm.tools["probe_session"] = probe
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(None, tm, cm)
    runner.execute_call("probe_session", {}, [])
    assert isinstance(seen["session"], SessionHandle)
    assert seen["session"].runner is runner
    assert seen["session"].context_manager is cm
    evt = [e for e in cm.trace.events if e["t"] == "tool_call"][0]
    assert evt["injected"] == []
    assert cm.get_context("current_turn_calls")[0]["args"] == {}


def test_tool_index_refresh_adds_and_removes_extra_cards(tmp_path):
    from core.skills import SkillCard
    from core.tool_index import ToolIndex
    idx = ToolIndex(persist_directory=str(tmp_path))
    base = idx.collection.count()
    card = SkillCard(name="skill:demo", description="Demo skill for tests.",
                     params={"freq": {"type": "number"}}, required=("freq",))
    idx.refresh([card])
    assert idx.collection.count() == base + 1
    assert any(c.name == "skill:demo" for c in idx.search("demo skill"))
    idx.refresh([])
    assert idx.collection.count() == base
