import numpy as np
import pytest
from core.tool_manager import ToolManager


@pytest.fixture
def tm():
    return ToolManager()


def test_schemas_are_openai_wrapped(tm):
    schemas = tm.get_tool_schemas()
    assert all(s["type"] == "function" for s in schemas)
    assert {s["function"]["name"] for s in schemas} >= {"make_ricker", "wedge_model"}


def test_unknown_tool_raises(tm):
    with pytest.raises(ValueError, match="Unknown tool"):
        tm.process_tool_call("does_not_exist", {})


def test_defaults_filled_for_make_ricker(tm):
    time_array, wavelet = tm.process_tool_call("make_ricker", {"frequency": 30})
    assert len(wavelet) > 10
    assert np.isfinite(wavelet).all()


def test_missing_required_raises(tm):
    with pytest.raises(ValueError, match="required"):
        tm.process_tool_call("make_ricker", {})


def test_no_setdefault_attribute_drift(tm):
    from core.tool_registry import REGISTRY_BY_NAME
    assert set(tm.tools.keys()) == set(REGISTRY_BY_NAME.keys())
