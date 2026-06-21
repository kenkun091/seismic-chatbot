import os
import numpy as np
import pytest

from workflows.recipes.eei_optimal_chi_petro import eei_optimal_chi_petro


def _petro_logs(n=40, seed=0):
    rng = np.random.RandomState(seed)
    phit = 0.05 + 0.25 * rng.rand(n)
    vclay = 0.50 * rng.rand(n)
    return phit.tolist(), vclay.tolist()


def test_petro_recipe_vclay_target():
    phit, vclay = _petro_logs()
    res = eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine")
    assert {"optimal_chi", "max_correlation", "chi", "correlation",
            "eei_optimal", "target", "image_path"} <= set(res)
    assert res["target"] == "vclay"
    assert -90.0 <= res["optimal_chi"] <= 90.0
    assert abs(res["max_correlation"]) <= 1.0 + 1e-9
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_petro_recipe_phit_target_runs():
    phit, vclay = _petro_logs(seed=2)
    res = eei_optimal_chi_petro(phit, vclay, target="phit")
    assert res["target"] == "phit"
    os.remove(res["image_path"])


def test_petro_recipe_rejects_bad_target():
    phit, vclay = _petro_logs()
    with pytest.raises(ValueError):
        eei_optimal_chi_petro(phit, vclay, target="bogus")
