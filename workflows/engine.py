"""Workflow engine: declarative recipe registry + a run() entry point.

Each recipe is declared once as a frozen `WorkflowSpec`. `core/tool_registry.py`
converts these into `ToolSpec`s and appends them to the tool REGISTRY, so the
chatbot runs a workflow exactly like any other tool. `WorkflowEngine.run` is the
programmatic / future-sweep entry point that fills defaults, checks required
params, and calls the recipe.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from workflows.recipes.petro_to_avo import petro_to_avo


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    fn: Callable
    description: str
    params: dict
    required: list
    defaults: dict = field(default_factory=dict)
    auto_plot: Optional[str] = None


WORKFLOW_REGISTRY = [
    WorkflowSpec(
        name="petro_to_avo",
        fn=petro_to_avo,
        description=(
            "End-to-end AVO feasibility: predict elastic properties (Vp, Vs, density) "
            "of a sand and an overlying shale from porosity and clay volume (Han 1986), "
            "build the shale-over-sand interface, and model the AVO reflectivity curve "
            "and interpretation attributes (intercept A, gradient B, AVO class). Returns "
            "the two layers, the reflectivity-vs-angle curve, the AVO attributes, and a "
            "composite plot."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees for the AVO curve."},
            "fluid_sand": {"type": "string", "description": "Sand pore fluid: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "fluid_shale": {"type": "string", "description": "Shale pore fluid (default 'water')."},
            "method": {"type": "string", "description": "Reflectivity method: 'shuey' (default) or 'zoeppritz'."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"],
        defaults={"fluid_sand": "brine", "fluid_shale": "water", "method": "shuey"},
        auto_plot=None,
    ),
]

WORKFLOW_REGISTRY_BY_NAME = {w.name: w for w in WORKFLOW_REGISTRY}
WORKFLOW_NAMES = frozenset(WORKFLOW_REGISTRY_BY_NAME)


class WorkflowEngine:
    """Runs a registered workflow recipe by name (programmatic / sweep entry)."""

    def run(self, name, params):
        spec = WORKFLOW_REGISTRY_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"Unknown workflow: {name}")
        full = dict(spec.defaults)
        full.update(params)
        missing = [p for p in spec.required if p not in full]
        if missing:
            raise ValueError(f"{name}: missing required parameters: {missing}")
        return spec.fn(**full)
