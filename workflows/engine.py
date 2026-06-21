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
from workflows.recipes.saturation_sweep import saturation_sweep
from workflows.sweep import run_sweep


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
            "(Vclay for lithology, porosity, or water saturation Sw). Returns the optimal "
            "chi, the correlation-vs-chi curve, the EEI log at the optimal chi, and a plot."
        ),
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity log (fraction, 0-1)."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay-volume log (fraction, 0-1)."},
            "target": {"type": "string", "description": "Target property to correlate against: 'vclay' (default), 'phit', or 'sw'."},
            "fluid": {"type": "string", "description": "Pore fluid for the vclay/phit prediction: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "chi_min": {"type": "number", "description": "Minimum rotation angle in degrees (default -90)."},
            "chi_max": {"type": "number", "description": "Maximum rotation angle in degrees (default 90)."},
            "chi_step": {"type": "number", "description": "Rotation-angle step in degrees (default 1)."},
            "sw": {"type": "array", "items": {"type": "number"}, "description": "Water-saturation log (fraction, 0-1); required when target='sw'."},
            "hydrocarbon": {"type": "string", "description": "Hydrocarbon end-member for target='sw': 'gas' (default) or 'oil'."},
            "law": {"type": "string", "description": "Fluid-mixing law for target='sw': 'reuss' (default) or 'brie'."},
        },
        required=["phit", "vclay"],
        defaults={"target": "vclay", "fluid": "brine", "chi_min": -90.0, "chi_max": 90.0, "chi_step": 1.0, "sw": None, "hydrocarbon": "gas", "law": "reuss"},
        auto_plot=None,
    ),
    WorkflowSpec(
        name="saturation_sweep",
        fn=saturation_sweep,
        description=(
            "Saturation (fluid-line) analysis: for a single rock described by porosity "
            "and clay volume, compute Vp, Vs, acoustic impedance and Vp/Vs across a range "
            "of water saturations Sw, using an effective brine+hydrocarbon pore fluid "
            "mixed by the Reuss/Wood (uniform) or Brie (patchy) law. Returns the "
            "saturation curves and a plot. Useful for fluid feasibility / DHI sensitivity."
        ),
        params={
            "phit": {"type": "number", "description": "Porosity (fraction, 0-1)."},
            "vclay": {"type": "number", "description": "Clay volume (fraction, 0-1)."},
            "hydrocarbon": {"type": "string", "description": "Hydrocarbon end-member: 'gas' (default) or 'oil'."},
            "law": {"type": "string", "description": "Fluid-mixing law: 'reuss' (uniform/Wood, default) or 'brie' (patchy)."},
            "sw_values": {"type": "array", "items": {"type": "number"}, "description": "Water saturations to sweep (default 0 to 1 in 21 steps)."},
            "brie_exponent": {"type": "number", "description": "Brie exponent e (default 3); used only when law='brie'."},
        },
        required=["phit", "vclay"],
        defaults={"hydrocarbon": "gas", "law": "reuss", "sw_values": None, "brie_exponent": 3.0},
        auto_plot=None,
    ),
    WorkflowSpec(
        name="run_sweep",
        fn=run_sweep,
        description=(
            "Parameter sweep / sensitivity analysis: run another workflow recipe over a "
            "grid of parameter values (the cartesian product) and collect one scalar "
            "result metric per run. Returns a results table, summary statistics "
            "(min/max/mean/std for numeric metrics, or value counts for categorical ones "
            "like AVO class), a coverage report (which cells ran or failed), and an "
            "aggregate plot (a line for a 1-parameter sweep, a heatmap for two, a "
            "histogram otherwise). Use it to test how an output responds across ranges of "
            "porosity, clay volume, fluid, saturation, frequency, etc."
        ),
        params={
            "recipe": {"type": "string", "description": "Name of the workflow recipe to sweep, e.g. 'petro_to_avo', 'fluid_scenario', 'tuning', or 'saturation_sweep'."},
            "grid": {"type": "object", "description": "Swept parameters mapped to lists of values, e.g. {\"phit_sand\": [0.1, 0.2, 0.3], \"fluid_sand\": [\"brine\", \"gas\"]}. The cartesian product of these is run."},
            "metric": {"type": "string", "description": "Name of the scalar field in the recipe's result to collect per run, e.g. 'gradient', 'intercept', 'avo_class', 'tuning_thickness', or 'resolution_limit'."},
            "fixed": {"type": "object", "description": "Parameters held constant across every run (the recipe's other required/optional params)."},
        },
        required=["recipe", "grid", "metric"],
        defaults={"fixed": None},
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
