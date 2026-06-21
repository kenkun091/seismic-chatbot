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
