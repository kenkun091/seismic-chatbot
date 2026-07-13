import numpy as np

from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_rock_properties_saturation_registered():
    assert "rock_properties_saturation" in reg.REGISTRY_BY_NAME
    assert "rock_properties_saturation" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"rock_properties_saturation"}


def test_rock_properties_saturation_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("rock_properties_saturation", {
        "phit": [0.25], "vclay": [0.20], "sw": [0.5],
    })
    # tuple (vp, vs, rhob, vp_vs, ai, si)
    assert len(res) == 6
    vp = np.asarray(res[0], dtype=float)
    assert np.all(vp > 0)
