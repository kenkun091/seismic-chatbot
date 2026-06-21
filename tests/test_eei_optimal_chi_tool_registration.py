import numpy as np

from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_eei_optimal_chi_registered():
    assert "eei_optimal_chi" in reg.REGISTRY_BY_NAME
    assert "eei_optimal_chi" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"eei_optimal_chi"}
    assert reg.REGISTRY_BY_NAME["eei_optimal_chi"].auto_plot is None  # self-plots


def test_eei_optimal_chi_runs_through_tool_manager():
    # Randomized (non-degenerate) logs: with perfectly linear logs and vs=vp/2 the
    # EEI is near-affine in AI at every chi, so |r|=1 ties and argmax picks chi_min.
    rng = np.random.RandomState(0)
    n = 30
    vp = (3000.0 + 800.0 * rng.rand(n))
    vs = vp / 2.0 + 50.0 * rng.rand(n)   # keeps 0 < vs < vp
    rho = 2.2 + 0.3 * rng.rand(n)
    target = vp * rho  # acoustic impedance -> EEI peaks at chi=0
    vp, vs, rho, target = vp.tolist(), vs.tolist(), rho.tolist(), target.tolist()
    tm = ToolManager()
    res = tm.execute_tool("eei_optimal_chi", {
        "vp": vp, "vs": vs, "rho": rho, "target": target,
    })
    assert abs(res["optimal_chi"]) <= 1.0
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
