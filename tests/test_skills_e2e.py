import pytest

from core.skills import SkillRegistry, get_registry, set_registry


def test_builtin_ricker_skill_loads_from_repo():
    set_registry(None)
    reg = get_registry()
    try:
        skill = reg.get("ricker_wavelet")
        assert skill.source == "repo" and skill.chain[0]["tool"] == "make_ricker"
    finally:
        set_registry(None)


def test_builtin_skill_is_discoverable(tmp_path):
    from core.tool_index import ToolIndex
    reg = SkillRegistry(runtime_dir=str(tmp_path / "rt"))
    idx = ToolIndex(persist_directory=str(tmp_path))
    idx.refresh(reg.specs())
    names = [c.name for c in idx.search("build a ricker wavelet skill")]
    assert "skill:ricker_wavelet" in names


def test_orchestrator_can_run_skill_via_run_task(tmp_path):
    from core.orchestrator import SeismicOrchestrator
    from core.tool_manager import ToolManager

    class FakeFunc:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, name, arguments, call_id="c1"):
            self.id = call_id
            self.function = FakeFunc(name, arguments)

    class FakeLLM:
        def __init__(self, responses):
            self._responses = list(responses)

        def get_completion(self, *a, **k):
            return self._responses.pop(0)

    set_registry(SkillRegistry(runtime_dir=str(tmp_path / "rt")))
    try:
        llm = FakeLLM([
            {"content": "", "tool_calls": [FakeToolCall(
                "run_task", '{"brief": "call run_skill ricker_wavelet with frequency 40", '
                            '"tool_names": ["run_skill"]}')]},
            {"content": "", "tool_calls": [FakeToolCall(
                "run_skill", '{"name": "ricker_wavelet", "params": {"frequency": 40}}')]},
            {"content": "<reply>ran the skill</reply>", "tool_calls": None},
            {"content": "<reply>Done: 40 Hz wavelet</reply>", "tool_calls": None},
        ])
        orch = SeismicOrchestrator(llm_client=llm, tool_manager=ToolManager(),
                                   knowledge_base=object(), tool_index=object())
        orch.context_manager.trace.persist_dir = ""
        out = orch.process_single_input("run the ricker skill at 40 Hz")
        assert out["reply"] == "Done: 40 Hz wavelet"
        assert out["images"] and out["images"][0].endswith(".png")
        kinds = [e["t"] for e in out["trace"]["events"]]
        assert "skill_run" in kinds
        assert orch.context_manager.get_context("last_ricker_wavelet")["parameters"]["frequency"] == 40
    finally:
        set_registry(None)
