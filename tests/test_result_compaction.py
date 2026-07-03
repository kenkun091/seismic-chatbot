"""Unit tests for tool-result compaction: what the LLM sees in role:"tool"
messages. Large numeric arrays become summary strings; image paths are masked
(plots are shown to the user directly, the model must not echo them)."""
import numpy as np

from core.chatbot_tool_use import SeismicChatBotToolUse


def _bare_bot():
    # Build an instance WITHOUT running __init__ (avoids loading RAG/LLM deps).
    return object.__new__(SeismicChatBotToolUse)


def test_long_numeric_list_is_summarized():
    bot = _bare_bot()
    out = bot._compact_value(list(range(61)))
    assert out == "<61 values, min=0, max=60, first=0, last=60>"


def test_short_list_kept_verbatim():
    bot = _bare_bot()
    assert bot._compact_value([1, 2, 3]) == [1, 2, 3]


def test_large_ndarray_summarized_with_shape():
    bot = _bare_bot()
    out = bot._compact_value(np.zeros((100, 61)))
    assert out == "<array shape (100, 61), min=0, max=0>"


def test_small_ndarray_becomes_list():
    bot = _bare_bot()
    assert bot._compact_value(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_image_path_masked():
    bot = _bare_bot()
    out = bot._compact_value({"tuning_thickness": 12.5, "image_path": "/tmp/x.png"})
    assert out == {"tuning_thickness": 12.5,
                   "image_path": "<plot generated and shown to the user>"}


def test_nested_dict_recursed():
    bot = _bare_bot()
    out = bot._compact_value({"cases": {"gas": {"rc": [0.1] * 20}}})
    assert out["cases"]["gas"]["rc"].startswith("<20 values")


def test_scalars_strings_and_bools_passthrough():
    bot = _bare_bot()
    assert bot._compact_value(3.5) == 3.5
    assert bot._compact_value("AVO class III") == "AVO class III"
    assert bot._compact_value(True) is True
    assert bot._compact_value(None) is None


def test_bool_list_not_treated_as_numeric():
    bot = _bare_bot()
    assert bot._compact_value([True] * 20) == [True] * 20


def test_tuple_of_arrays_compacted():
    bot = _bare_bot()
    time_array = np.linspace(-0.1, 0.1, 201)
    wavelet = np.zeros(201)
    out = bot._compact_value((time_array, wavelet))
    assert isinstance(out, list) and len(out) == 2
    assert out[0].startswith("<array shape (201,)")


def test_compact_tool_result_returns_string():
    bot = _bare_bot()
    s = bot._compact_tool_result({"a": 1, "rc": list(range(20))})
    assert isinstance(s, str)
    assert '"a": 1' in s
    assert "<20 values" in s
