import json

from core.provenance import write_plot_provenance


def test_write_sidecar_next_to_artifact(tmp_path):
    png = tmp_path / "plot.png"
    png.write_bytes(b"fake")
    sidecar = write_plot_provenance(str(png), {"tool": "plot_ricker",
                                               "session": "s1", "turn": 2,
                                               "parameters": {"frequency": 30}})
    assert sidecar == str(png) + ".prov.json"
    data = json.loads((tmp_path / "plot.png.prov.json").read_text())
    assert data["artifact"] == "plot.png"
    assert data["generator"] == "seismic-chatbot"
    assert data["tool"] == "plot_ricker"
    assert data["parameters"] == {"frequency": 30}
    assert "created" in data


def test_write_failure_is_swallowed():
    assert write_plot_provenance("/dev/null/nope/plot.png", {"tool": "x"}) is None


def test_loop_writes_sidecar_for_auto_plotted_ricker():
    from core.context_manager import ContextManager
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

    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.session_id = "prov-session"
    cm.trace.begin_turn("30 Hz ricker")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker",
                                                    '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), ToolManager(), cm)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["images"], "auto-plot should have produced a png"
    sidecar_path = out["images"][0] + ".prov.json"
    data = json.loads(open(sidecar_path).read())
    assert data["session"] == "prov-session"
    assert data["turn"] == 1
    assert data["compute_tool"] == "make_ricker"
    assert data["compute_parameters"]["frequency"] == 30
    assert data["tool"] == "plot_ricker"


def test_write_provenance_never_raises_on_uncompactable_input():
    import numpy as np

    from core.context_manager import ContextManager
    from core.tool_loop import ToolLoopRunner

    cm = ContextManager()
    cm.trace.persist_dir = ""
    runner = ToolLoopRunner(None, None, cm)
    # non-numeric ndarray > 12 elements: compact_value raises on .max()
    poisoned = {"grid": np.array(["a", "b"] * 10)}
    runner._write_provenance(["/tmp/does-not-matter.png"], "t", poisoned)  # must not raise


def test_existing_sidecar_is_not_overwritten(tmp_path):
    png = tmp_path / "plot.png"
    png.write_bytes(b"x")
    write_plot_provenance(str(png), {"tool": "first"})
    write_plot_provenance(str(png), {"tool": "second"})
    assert json.loads((tmp_path / "plot.png.prov.json").read_text())["tool"] == "first"
