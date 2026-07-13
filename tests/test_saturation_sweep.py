import os
import numpy as np
import pytest

from workflows.recipes.saturation_sweep import saturation_sweep


def test_saturation_sweep_structure_and_plot():
    res = saturation_sweep(0.25, 0.20, hydrocarbon="gas", law="reuss")
    assert {"sw", "vp", "vs", "ai", "vp_vs", "hydrocarbon", "law", "image_path"} <= set(res)
    n = len(res["sw"])
    assert len(res["vp"]) == n == len(res["ai"]) > 1
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_saturation_sweep_vp_increases_with_sw():
    # Reuss: Vp at full brine (Sw=1) exceeds Vp at full gas (Sw=0).
    res = saturation_sweep(0.25, 0.20, sw_values=[0.0, 1.0])
    assert res["vp"][1] > res["vp"][0]


def test_saturation_sweep_rejects_bad_law():
    with pytest.raises(ValueError):
        saturation_sweep(0.25, 0.20, law="bogus")


def test_saturation_sweep_brie_law_runs():
    res = saturation_sweep(0.25, 0.20, law="brie", brie_exponent=3.0)
    assert res["law"] == "brie"
    assert len(res["vp"]) == len(res["sw"]) > 1
    os.remove(res["image_path"])
