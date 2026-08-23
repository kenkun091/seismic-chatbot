"""Registry contract for the outcrop tools."""
from core import tool_registry as reg
from core.tool_manager import ToolManager

NEW = ["interpret_outcrop", "plot_outcrop_interpretation", "outcrop_to_model",
       "synthetic_section", "plot_seismic_section"]


def test_tools_registered_with_functions():
    for name in NEW:
        assert name in reg.REGISTRY_BY_NAME, name
        assert callable(reg.TOOL_FUNCTIONS[name])


def test_auto_plot_chain():
    assert reg.AUTO_PLOT["interpret_outcrop"] == "plot_outcrop_interpretation"
    assert reg.AUTO_PLOT["synthetic_section"] == "plot_seismic_section"
    assert "outcrop_to_model" not in reg.AUTO_PLOT


def test_context_filled_params_are_optional_with_none_default():
    spec = reg.REGISTRY_BY_NAME
    assert "image_path" not in spec["interpret_outcrop"].required
    assert spec["interpret_outcrop"].defaults["image_path"] is None
    assert "interpretation" not in spec["outcrop_to_model"].required
    assert spec["outcrop_to_model"].defaults["interpretation"] is None
    assert "model" not in spec["synthetic_section"].required
    assert spec["synthetic_section"].defaults["model"] is None


def test_schema_descriptions_tell_llm_not_to_pass_context_params():
    schemas = {s["name"]: s for s in reg.TOOL_SCHEMAS}
    assert "automatically" in schemas["interpret_outcrop"]["parameters"]["properties"]["image_path"]["description"].lower()
    assert "automatically" in schemas["outcrop_to_model"]["parameters"]["properties"]["interpretation"]["description"].lower()
    assert "automatically" in schemas["synthetic_section"]["parameters"]["properties"]["model"]["description"].lower()


def test_tool_manager_surfaces_clear_errors_without_context():
    tm = ToolManager()
    import pytest
    with pytest.raises(ValueError, match="upload an outcrop photo"):
        tm.process_tool_call("interpret_outcrop", {})
    with pytest.raises(ValueError, match="interpret_outcrop"):
        tm.process_tool_call("outcrop_to_model", {"height_m": 10})
    with pytest.raises(ValueError, match="earth model first"):
        tm.process_tool_call("synthetic_section", {})


def test_synthetic_section_defaults():
    d = reg.REGISTRY_BY_NAME["synthetic_section"].defaults
    assert d["dt"] == 1.0 and d["domain"] == "time" and d["method"] == "shuey"
