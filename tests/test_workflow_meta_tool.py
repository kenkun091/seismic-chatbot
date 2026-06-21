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


def test_tuning_is_registered_meta_tool():
    assert "tuning" in reg.REGISTRY_BY_NAME
    assert "tuning" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"tuning"}


def test_tuning_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("tuning", {
        "phit_sand": 0.28, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "max_thickness": 40.0,
    })
    assert res["tuning_thickness"] > 0
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_eei_optimal_chi_petro_is_registered_meta_tool():
    assert "eei_optimal_chi_petro" in reg.REGISTRY_BY_NAME
    assert "eei_optimal_chi_petro" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"eei_optimal_chi_petro"}


def test_eei_optimal_chi_petro_runs_through_tool_manager():
    import numpy as np
    n = 30
    phit = list(0.10 + 0.002 * np.arange(n))
    vclay = list(0.10 + 0.01 * np.arange(n))
    tm = ToolManager()
    res = tm.execute_tool("eei_optimal_chi_petro", {"phit": phit, "vclay": vclay})
    assert res["target"] == "vclay"  # default
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_saturation_sweep_is_registered_meta_tool():
    assert "saturation_sweep" in reg.REGISTRY_BY_NAME
    assert "saturation_sweep" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"saturation_sweep"}


def test_saturation_sweep_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("saturation_sweep", {"phit": 0.25, "vclay": 0.20})
    assert res["law"] == "reuss"  # default
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_run_sweep_is_registered_meta_tool():
    assert "run_sweep" in reg.REGISTRY_BY_NAME
    assert "run_sweep" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"run_sweep"}


def test_run_sweep_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("run_sweep", {
        "recipe": "petro_to_avo",
        "grid": {"fluid_sand": ["brine", "gas"]},
        "metric": "gradient",
        "fixed": {"phit_sand": 0.25, "vclay_sand": 0.15,
                  "phit_shale": 0.10, "vclay_shale": 0.55,
                  "angles": [0, 10, 20, 30]},
    })
    assert res["coverage"]["ran"] == 2
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
