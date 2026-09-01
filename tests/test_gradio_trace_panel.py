from interfaces.gradio_interface import append_bot_response

_TRACE = {"session": "s", "turn": 1, "tools_used": ["make_ricker"], "events": [
    {"t": "turn_start", "ts": 1.0, "input": "x"},
    {"t": "physics_warning", "ts": 1.1, "tool": "make_ricker",
     "category": "UserWarning", "message": "vp outside typical range"},
]}


def test_flags_are_appended_to_the_bubble():
    history = append_bot_response([["hi", None]],
                                  {"reply": "done", "images": [], "trace": _TRACE})
    assert history[-1][1].startswith("done")
    assert "⚠️ Physics: vp outside typical range" in history[-1][1]


def test_no_flags_leaves_reply_untouched():
    quiet = {"session": "s", "turn": 1, "tools_used": [], "events": [
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "intent", "ts": 1.1, "verdict": "KNOWLEDGE", "via": "llm"},
    ]}
    history = append_bot_response([["hi", None]],
                                  {"reply": "done", "images": [], "trace": quiet})
    assert history[-1][1] == "done"


def test_traceless_response_unchanged():
    history = append_bot_response([["hi", None]], {"reply": "done", "images": []})
    assert history[-1][1] == "done"


def test_plain_string_response_still_renders():
    history = append_bot_response([["hi", None]], "plain")
    assert history[-1][1] == "plain"


def test_summary_failure_degrades_to_flagless(monkeypatch):
    import interfaces.gradio_interface as gi

    def boom(trace):
        raise RuntimeError("summary broke")

    monkeypatch.setattr(gi, "summarize_trace", boom)
    history = gi.append_bot_response([["hi", None]],
                                     {"reply": "done", "images": [], "trace": _TRACE})
    assert history[-1][1] == "done"
