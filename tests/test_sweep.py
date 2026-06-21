import os
import numpy as np
import pytest

from workflows.sweep import _expand_grid


def test_expand_grid_cartesian_product():
    combos = _expand_grid({"a": [1, 2, 3], "b": ["x", "y"]})
    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos
    assert {"a": 3, "b": "y"} in combos


def test_expand_grid_merges_fixed():
    combos = _expand_grid({"a": [1, 2]}, fixed={"c": 9})
    assert combos == [{"c": 9, "a": 1}, {"c": 9, "a": 2}]


def test_expand_grid_swept_overrides_fixed():
    combos = _expand_grid({"a": [5]}, fixed={"a": 0, "c": 9})
    assert combos == [{"c": 9, "a": 5}]


def test_expand_grid_rejects_empty_grid():
    with pytest.raises(ValueError):
        _expand_grid({})


def test_expand_grid_rejects_empty_value_list():
    with pytest.raises(ValueError):
        _expand_grid({"a": [1], "b": []})


from workflows.sweep import _summarize


def test_summarize_numeric():
    s = _summarize([1.0, 2.0, 3.0, None])
    assert s["kind"] == "numeric"
    assert s["count"] == 3
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert np.isclose(s["mean"], 2.0)
    assert np.isclose(s["std"], np.std([1.0, 2.0, 3.0]))


def test_summarize_categorical():
    s = _summarize(["III", "III", "IIp", None])
    assert s["kind"] == "categorical"
    assert s["count"] == 3
    assert s["counts"] == {"III": 2, "IIp": 1}


def test_summarize_empty():
    s = _summarize([None, None])
    assert s["kind"] == "empty"
    assert s["count"] == 0


def test_summarize_bool_is_categorical():
    # bool must NOT be treated as numeric (True/False are categorical here).
    s = _summarize([True, False, True])
    assert s["kind"] == "categorical"
    assert s["counts"] == {True: 2, False: 1}


from workflows.sweep import run_sweep

# Fixed params that make petro_to_avo run for any fluid_sand value.
_PETRO_FIXED = {
    "phit_sand": 0.25, "vclay_sand": 0.15,
    "phit_shale": 0.10, "vclay_shale": 0.55,
    "angles": [0, 10, 20, 30],
}


def test_run_sweep_1d_numeric_metric():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert res["recipe"] == "petro_to_avo"
    assert res["metric"] == "gradient"
    assert res["swept_params"] == ["fluid_sand"]
    assert len(res["rows"]) == 2
    assert all(isinstance(r["value"], float) for r in res["rows"])
    assert res["coverage"] == {"total": 2, "ran": 2, "failed": 0, "failures": []}
    assert res["stats"]["kind"] == "numeric"


def test_run_sweep_2d_grid_cell_count():
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.20, 0.30], "fluid_sand": ["brine", "gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert len(res["rows"]) == 4
    assert res["coverage"]["ran"] == 4


def test_run_sweep_categorical_metric():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="avo_class", fixed=_PETRO_FIXED)
    assert res["stats"]["kind"] == "categorical"
    assert sum(res["stats"]["counts"].values()) == 2


def test_run_sweep_records_failed_cell_without_aborting():
    # phit_sand=1.5 is non-physical -> that cell raises; the brine/gas valid
    # cells still run. (Grid value overrides the fixed phit_sand.)
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.25, 1.5], "fluid_sand": ["gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert res["coverage"]["total"] == 2
    assert res["coverage"]["ran"] == 1
    assert res["coverage"]["failed"] == 1
    assert len(res["coverage"]["failures"]) == 1


def test_run_sweep_rejects_self():
    with pytest.raises(ValueError):
        run_sweep("run_sweep", {"a": [1]}, metric="x")


def test_run_sweep_rejects_unknown_recipe():
    with pytest.raises(ValueError):
        run_sweep("not_a_recipe", {"a": [1]}, metric="x")


def test_run_sweep_all_cells_fail_raises():
    # A bad metric makes every cell fail -> ValueError naming the problem.
    with pytest.raises(ValueError):
        run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                  metric="nonexistent_metric", fixed=_PETRO_FIXED)


def test_run_sweep_cleans_up_cell_pngs(monkeypatch):
    removed = []
    real_remove = os.remove
    monkeypatch.setattr("workflows.sweep.os.remove",
                        lambda p: (removed.append(p), real_remove(p)))
    run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
              metric="gradient", fixed=_PETRO_FIXED)
    # Two cells -> two per-cell PNGs deleted.
    assert len([p for p in removed if p.endswith(".png")]) == 2


def _png_ok(path):
    return isinstance(path, str) and path.endswith(".png") and os.path.getsize(path) > 0


def test_run_sweep_adds_image_path_1d():
    res = run_sweep("petro_to_avo", {"phit_sand": [0.15, 0.25, 0.35]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])


def test_run_sweep_adds_image_path_2d_heatmap():
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.15, 0.25], "vclay_sand": [0.10, 0.20]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])


def test_run_sweep_adds_image_path_categorical():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="avo_class", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])


def test_run_sweep_adds_image_path_histogram_3d():
    # >2 swept numeric dims -> the histogram fallback branch of plot_sweep.
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.20, 0.30], "vclay_sand": [0.10, 0.20],
                     "phit_shale": [0.08, 0.12]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert len(res["rows"]) == 8
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])
