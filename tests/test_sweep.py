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
