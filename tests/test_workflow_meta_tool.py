from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_petro_to_avo_is_registered_meta_tool():
    assert "petro_to_avo" in reg.REGISTRY_BY_NAME
    assert "petro_to_avo" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"petro_to_avo"}
    spec = reg.REGISTRY_BY_NAME["petro_to_avo"]
    assert spec.auto_plot is None  # the recipe plots itself


def test_petro_to_avo_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("petro_to_avo", {
        "phit_sand": 0.25, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert isinstance(res, dict)
    assert res["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_fluid_scenario_is_registered_meta_tool():
    assert "fluid_scenario" in reg.REGISTRY_BY_NAME
    assert "fluid_scenario" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"fluid_scenario"}


def test_fluid_scenario_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("fluid_scenario", {
        "phit_sand": 0.28, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert set(res["cases"]) == {"brine", "gas"}  # default fluids
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
