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
from workflows.recipes.fluid_scenario import fluid_scenario
from workflows.recipes.tuning import tuning
from workflows.recipes.eei_optimal_chi_petro import eei_optimal_chi_petro


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
    WorkflowSpec(
        name="fluid_scenario",
        fn=fluid_scenario,
        description=(
            "AVO fluid-substitution scenarios: predict an in-situ sand and overlying "
            "shale from porosity and clay volume, then use Gassmann fluid substitution "
            "to model and compare the AVO response (reflectivity curve, intercept, "
            "gradient, AVO class) for each pore fluid (e.g. brine vs gas). Returns the "
            "per-fluid results and an overlaid comparison plot. Useful for DHI / "
            "fluid-feasibility assessment."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees for the AVO curves."},
            "fluids": {"type": "array", "items": {"type": "string"}, "description": "Pore fluids to compare, e.g. ['brine','gas'] (default). Each is 'brine'/'water', 'oil', or 'gas'."},
            "fluid_in": {"type": "string", "description": "In-situ pore fluid the sand is predicted at before substitution (default 'brine')."},
            "method": {"type": "string", "description": "Reflectivity method: 'shuey' (default) or 'zoeppritz'."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"],
        defaults={"fluids": None, "fluid_in": "brine", "method": "shuey"},
        auto_plot=None,
    ),
    WorkflowSpec(
        name="tuning",
        fn=tuning,
        description=(
            "Wedge tuning / vertical-resolution analysis: predict a sand and encasing "
            "shale from porosity and clay volume, build a sand wedge between two shale "
            "layers, and analyze the amplitude-vs-thickness response for the tuning "
            "thickness and resolution limit at a given wavelet frequency. Returns the "
            "tuning thickness, resolution limit, the amplitude-vs-thickness curve, and "
            "a tuning-curve plot."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "max_thickness": {"type": "number", "description": "Maximum wedge thickness in meters."},
            "wavelet_freq": {"type": "number", "description": "Ricker wavelet dominant frequency in Hz (default 30)."},
            "num_traces": {"type": "integer", "description": "Number of thickness traces across the wedge (default 61)."},
            "fluid_sand": {"type": "string", "description": "Sand pore fluid: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "max_thickness"],
        defaults={"wavelet_freq": 30.0, "num_traces": 61, "fluid_sand": "brine"},
        auto_plot=None,
    ),
    WorkflowSpec(
        name="eei_optimal_chi_petro",
        fn=eei_optimal_chi_petro,
        description=(
            "EEI optimal-rotation-angle analysis from petrophysics: predict Vp/Vs/density "
            "logs from porosity and clay-volume logs, then find the Extended Elastic "
            "Impedance angle chi whose EEI log best correlates with a chosen target "
            "(Vclay for lithology, or porosity). Returns the optimal chi, the "
            "correlation-vs-chi curve, the EEI log at the optimal chi, and a plot."
        ),
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity log (fraction, 0-1)."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay-volume log (fraction, 0-1)."},
            "target": {"type": "string", "description": "Target property to correlate against: 'vclay' (default) or 'phit'."},
            "fluid": {"type": "string", "description": "Pore fluid for the rock-physics prediction: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "chi_min": {"type": "number", "description": "Minimum rotation angle in degrees (default -90)."},
            "chi_max": {"type": "number", "description": "Maximum rotation angle in degrees (default 90)."},
            "chi_step": {"type": "number", "description": "Rotation-angle step in degrees (default 1)."},
        },
        required=["phit", "vclay"],
        defaults={"target": "vclay", "fluid": "brine", "chi_min": -90.0, "chi_max": 90.0, "chi_step": 1.0},
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
