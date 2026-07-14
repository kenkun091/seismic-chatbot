import pytest
from core import tool_registry as reg


def test_registry_nonempty():
    assert len(reg.REGISTRY) == 32


def test_names_unique():
    names = [s.name for s in reg.REGISTRY]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


def test_required_subset_of_params():
    for s in reg.REGISTRY:
        missing = set(s.required) - set(s.params.keys())
        assert not missing, f"{s.name}: required not in params: {missing}"


def test_defaults_subset_of_params():
    for s in reg.REGISTRY:
        missing = set(s.defaults.keys()) - set(s.params.keys())
        assert not missing, f"{s.name}: defaults not in params: {missing}"


def test_fn_callable():
    for s in reg.REGISTRY:
        assert callable(s.fn), f"{s.name}: fn not callable"


def test_auto_plot_resolves():
    names = {s.name for s in reg.REGISTRY}
    for s in reg.REGISTRY:
        if s.auto_plot is not None:
            assert s.auto_plot in names, f"{s.name}: auto_plot '{s.auto_plot}' unknown"


def test_derived_views_cover_registry():
    names = {s.name for s in reg.REGISTRY}
    assert {t["name"] for t in reg.TOOL_SCHEMAS} == names
    assert set(reg.TOOL_FUNCTIONS.keys()) == names
    assert set(reg.REGISTRY_BY_NAME.keys()) == names


def test_openai_schema_shape():
    s = reg.to_openai_schema(reg.REGISTRY[0])
    assert set(s.keys()) == {"name", "description", "parameters"}
    assert s["parameters"]["type"] == "object"
    assert "properties" in s["parameters"]


def test_tool_schemas_module_reexports_registry():
    from config import tool_schemas
    from core import tool_registry as reg
    assert tool_schemas.TOOL_SCHEMAS is reg.TOOL_SCHEMAS
    assert tool_schemas.TOOL_FUNCTIONS is reg.TOOL_FUNCTIONS
