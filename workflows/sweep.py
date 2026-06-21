"""Generic parameter-sweep engine for workflow recipes (gap S3).

run_sweep runs a registered recipe over the cartesian product of a parameter
grid, collects a chosen scalar metric per cell, and returns a results table,
summary statistics, coverage, and an aggregate plot. Per-cell recipe plots are
deleted; one aggregate plot is produced. A cell whose recipe raises is recorded
(value None) and the sweep continues.
"""
import itertools
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt


def _expand_grid(grid, fixed=None):
    """Cartesian product of a {param: [values]} grid, each merged with `fixed`.

    Returns a list of full parameter dicts. Swept keys override `fixed` on
    collision. Raises ValueError on an empty grid or an empty value list.
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty {param: [values]} mapping")
    fixed = dict(fixed or {})
    keys = list(grid)
    value_lists = []
    for k in keys:
        vals = grid[k]
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            raise ValueError(f"grid['{k}'] must be a non-empty list of values")
        value_lists.append(list(vals))

    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(fixed)
        combo.update(dict(zip(keys, values)))
        combos.append(combo)
    return combos
