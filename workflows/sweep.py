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


def _is_number(x):
    """True for real numeric scalars, excluding bool (bool is treated categorically)."""
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _summarize(values):
    """Summarize swept metric values (numeric stats or categorical counts)."""
    present = [v for v in values if v is not None]
    if not present:
        return {"kind": "empty", "count": 0}
    if all(_is_number(v) for v in present):
        arr = np.asarray([float(v) for v in present], dtype=float)
        return {
            "kind": "numeric",
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    return {"kind": "categorical", "count": len(present), "counts": counts}


def run_sweep(recipe, grid, metric, fixed=None):
    """Run `recipe` over the cartesian product of `grid`, collecting `metric` per cell.

    Returns a results table, summary stats, and coverage. Per-cell recipe PNGs are
    deleted. Cells whose recipe raises (or that lack `metric`) are recorded in
    coverage.failures with value None; the sweep does not abort. Raises ValueError
    for a self-sweep, an unknown recipe, a bad grid, or if every cell fails.
    """
    # Lazy import to avoid the engine<->sweep import cycle.
    from workflows.engine import WorkflowEngine, WORKFLOW_NAMES

    if recipe == "run_sweep":
        raise ValueError("cannot sweep run_sweep itself")
    if recipe not in WORKFLOW_NAMES:
        raise ValueError(f"unknown recipe {recipe!r}; choose one of {sorted(WORKFLOW_NAMES)}")

    swept_params = list(grid)
    combos = _expand_grid(grid, fixed)
    engine = WorkflowEngine()

    rows = []
    failures = []
    for combo in combos:
        swept_only = {k: combo[k] for k in swept_params}
        try:
            result = engine.run(recipe, combo)
        except Exception as exc:  # a recipe raised -> record, do not abort
            failures.append({"params": swept_only, "error": str(exc)})
            rows.append({"params": swept_only, "value": None})
            continue
        img = result.get("image_path")
        if isinstance(img, str) and os.path.exists(img):
            os.remove(img)
        if metric not in result:
            failures.append({"params": swept_only,
                             "error": f"metric {metric!r} not in result keys {sorted(result)}"})
            rows.append({"params": swept_only, "value": None})
            continue
        rows.append({"params": swept_only, "value": result[metric]})

    total = len(rows)
    ran = sum(1 for r in rows if r["value"] is not None)
    if ran == 0:
        first = failures[0]["error"] if failures else "no cells produced a value"
        raise ValueError(f"all {total} sweep cells failed; first error: {first}")

    out = {
        "recipe": recipe,
        "metric": metric,
        "swept_params": swept_params,
        "rows": rows,
        "stats": _summarize([r["value"] for r in rows]),
        "coverage": {"total": total, "ran": ran, "failed": total - ran,
                     "failures": failures},
    }
    out["image_path"] = plot_sweep(out)
    return out


def _numeric_param_values(rows, key):
    """Return the per-row float values of a swept param, or None if any is non-numeric."""
    vals = [r["params"][key] for r in rows]
    if all(_is_number(v) for v in vals):
        return [float(v) for v in vals]
    return None


def plot_sweep(result, output_path=None):
    """Aggregate plot for a sweep: line (1-D numeric), heatmap (2-D numeric),
    histogram (other numeric), or bar of counts (categorical metric)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    rows = result["rows"]
    metric = result["metric"]
    swept = result["swept_params"]
    numeric_metric = result["stats"]["kind"] == "numeric"

    fig, ax = plt.subplots(figsize=(9, 6))

    if not numeric_metric:
        # Categorical metric: bar of value counts.
        counts = result["stats"].get("counts", {})
        labels = [str(k) for k in counts]
        ax.bar(labels, [counts[k] for k in counts], color="C0")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"{result['recipe']}: {metric} distribution")
    elif len(swept) == 1 and _numeric_param_values(rows, swept[0]) is not None:
        x = _numeric_param_values(rows, swept[0])
        y = [r["value"] if r["value"] is not None else np.nan for r in rows]
        order = np.argsort(x)
        ax.plot(np.asarray(x)[order], np.asarray(y, dtype=float)[order], "o-", color="C0")
        ax.set_xlabel(swept[0])
        ax.set_ylabel(metric)
        ax.set_title(f"{result['recipe']}: {metric} vs {swept[0]}")
        ax.grid(True, alpha=0.3)
    elif (len(swept) == 2 and _numeric_param_values(rows, swept[0]) is not None
          and _numeric_param_values(rows, swept[1]) is not None):
        xs = sorted({r["params"][swept[0]] for r in rows})
        ys = sorted({r["params"][swept[1]] for r in rows})
        grid_z = np.full((len(ys), len(xs)), np.nan)
        for r in rows:
            i = ys.index(r["params"][swept[1]])
            j = xs.index(r["params"][swept[0]])
            grid_z[i, j] = r["value"] if r["value"] is not None else np.nan
        im = ax.imshow(grid_z, origin="lower", aspect="auto",
                       extent=[min(xs), max(xs), min(ys), max(ys)])
        fig.colorbar(im, ax=ax, label=metric)
        ax.set_xlabel(swept[0])
        ax.set_ylabel(swept[1])
        ax.set_title(f"{result['recipe']}: {metric}")
    else:
        # >2 swept dims or non-numeric params: distribution histogram.
        vals = [r["value"] for r in rows if r["value"] is not None]
        ax.hist(vals, bins=min(20, max(5, len(vals))), color="C0")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"{result['recipe']}: {metric} distribution ({len(swept)} params)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
