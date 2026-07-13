import pytest

from workflows.engine import (
    WorkflowSpec, WORKFLOW_REGISTRY, WORKFLOW_REGISTRY_BY_NAME,
    WORKFLOW_NAMES, WorkflowEngine,
)


def test_registry_contains_petro_to_avo():
    assert "petro_to_avo" in WORKFLOW_REGISTRY_BY_NAME
    assert "petro_to_avo" in WORKFLOW_NAMES
    spec = WORKFLOW_REGISTRY_BY_NAME["petro_to_avo"]
    assert isinstance(spec, WorkflowSpec)
    assert callable(spec.fn)
    assert set(spec.required) == {"phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"}
    assert spec.defaults == {"fluid_sand": "brine", "fluid_shale": "water", "method": "shuey"}


def test_run_fills_defaults_and_executes():
    eng = WorkflowEngine()
    res = eng.run("petro_to_avo", {
        "phit_sand": 0.25, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert res["method"] == "shuey"            # default filled
    assert res["lower"]["label"] == "sand"
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_run_unknown_workflow_raises():
    with pytest.raises(ValueError):
        WorkflowEngine().run("does_not_exist", {})


def test_run_missing_required_raises():
    with pytest.raises(ValueError):
        WorkflowEngine().run("petro_to_avo", {"phit_sand": 0.25})  # missing others
