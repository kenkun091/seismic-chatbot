"""
Single source of truth for the seismic chatbot's LLM-facing tools.

Every tool is declared once as a ToolSpec. The OpenAI/DeepSeek schemas,
the name->function map, the auto-plot chaining map, and the validation
wiring are all DERIVED from REGISTRY — nothing else is hand-maintained.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools.ricker_tools import create_ricker_wavelet, plot_wavelet
# TODO(Task 8): re-enable make_ormsby/analyze_wedge entries and real validators
# from tools.ricker_tools import create_ormsby_wavelet  # does not exist yet (Task 8)
from tools.wedge_tools import create_wedge_model, plot_wedge_model
# from tools.wedge_tools import analyze_wedge  # does not exist yet (Task 7/8)
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity
from tools.rock_physics_tools import calculate_rock_properties, rock_physics_rag
from tools.rag_tools import knowledge_rag
# TODO(Task 8): from tools.parameter_validation import validate_make_ricker, validate_wedge_model, validate_avo


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable
    description: str
    params: dict
    required: list
    defaults: dict = field(default_factory=dict)
    validator: Optional[Callable] = None
    auto_plot: Optional[str] = None


REGISTRY = [
    ToolSpec(
        name="make_ricker",
        fn=create_ricker_wavelet,
        description="Creates a Ricker wavelet with specified frequency and time parameters.",
        params={
            "frequency": {
                "type": "number",
                "description": "The dominant frequency of the Ricker wavelet in Hz (typically 10-100 Hz)."
            },
            "time_length": {
                "type": "number",
                "description": "Total length of the wavelet in milliseconds (default: 256 ms)."
            },
            "dt": {
                "type": "number",
                "description": "Time sampling interval in seconds (default: 0.001)."
            },
        },
        required=["frequency"],
        defaults={"time_length": 256.0, "dt": 0.001},
        validator=None,  # TODO(Task 8): validate_make_ricker
        auto_plot="plot_ricker",
    ),
    # TODO(Task 8): re-enable make_ormsby ToolSpec (fn=create_ormsby_wavelet, auto_plot="plot_ricker")
    ToolSpec(
        name="plot_ricker",
        fn=plot_wavelet,
        description="Plots a Ricker wavelet with time domain and frequency domain analysis.",
        params={
            "wavelet": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of wavelet amplitudes."
            },
            "time_array": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of time values in milliseconds."
            },
        },
        required=["wavelet"],
        defaults={},
    ),
    ToolSpec(
        name="wedge_model",
        fn=create_wedge_model,
        description="Creates a wedge model for seismic analysis with variable thickness layers.",
        params={
            "max_thickness": {
                "type": "number",
                "description": "Maximum thickness of the wedge layer in meters."
            },
            "v1": {
                "type": "number",
                "description": "P-wave velocity of the first layer in m/s."
            },
            "v2": {
                "type": "number",
                "description": "P-wave velocity of the second layer (wedge) in m/s."
            },
            "v3": {
                "type": "number",
                "description": "P-wave velocity of the third layer in m/s."
            },
            "rho1": {
                "type": "number",
                "description": "Density of the first layer in g/cm³."
            },
            "rho2": {
                "type": "number",
                "description": "Density of the second layer (wedge) in g/cm³."
            },
            "rho3": {
                "type": "number",
                "description": "Density of the third layer in g/cm³."
            },
            "wavelet_freq": {
                "type": "number",
                "description": "Frequency of the wavelet in Hz (default: 30 Hz)."
            },
            "num_traces": {
                "type": "integer",
                "description": "Number of traces in the wedge model (default: 61)."
            },
            "vs1": {
                "type": "number",
                "description": "S-wave velocity of the first layer in m/s (optional, defaults to v1/2)."
            },
            "vs2": {
                "type": "number",
                "description": "S-wave velocity of the second layer in m/s (optional, defaults to v2/2)."
            },
            "vs3": {
                "type": "number",
                "description": "S-wave velocity of the third layer in m/s (optional, defaults to v3/2)."
            },
            "incident_angle": {
                "type": "number",
                "description": "Incident angle in degrees for angle-dependent reflectivity calculation (default: 0)."
            },
            "export_path": {
                "type": "string",
                "description": "Optional path to write synthetic curves as CSV."
            },
        },
        required=["max_thickness", "v1", "v2", "v3", "rho1", "rho2", "rho3"],
        defaults={"wavelet_freq": 30.0, "num_traces": 61},
        validator=None,  # TODO(Task 8): validate_wedge_model
        auto_plot="plot_wedge_model",
    ),
    ToolSpec(
        name="plot_wedge_model",
        fn=plot_wedge_model,
        description="Plots a wedge model showing seismic response vs thickness.",
        params={
            "synthetic_data": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                },
                "description": "Array of synthetic seismic data."
            },
            "parameters": {
                "type": "object",
                "description": "Parameters used to create the wedge model."
            },
            "figsize": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Figure size as [width, height] in inches. Default is [12, 14]. Use larger values like [18, 21] for enlargement."
            },
        },
        required=["synthetic_data", "parameters"],
        defaults={},
    ),
    # TODO(Task 8): re-enable analyze_wedge ToolSpec (fn=analyze_wedge, required synthetic_data+parameters)
    ToolSpec(
        name="zoeppritz_reflectivity",
        fn=zoeppritz_reflectivity,
        description="Calculates reflectivity using the Zoeppritz equations for elastic wave reflection.",
        params={
            "vp1": {
                "type": "number",
                "description": "P-wave velocity of the first medium in m/s."
            },
            "vs1": {
                "type": "number",
                "description": "S-wave velocity of the first medium in m/s."
            },
            "rho1": {
                "type": "number",
                "description": "Density of the first medium in g/cm³."
            },
            "vp2": {
                "type": "number",
                "description": "P-wave velocity of the second medium in m/s."
            },
            "vs2": {
                "type": "number",
                "description": "S-wave velocity of the second medium in m/s."
            },
            "rho2": {
                "type": "number",
                "description": "Density of the second medium in g/cm³."
            },
            "angles": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of incidence angles in degrees."
            },
        },
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"],
        defaults={},
        validator=None,  # TODO(Task 8): validate_avo
        auto_plot="plot_avo_reflectivity",
    ),
    ToolSpec(
        name="shuey_reflectivity",
        fn=shuey_reflectivity,
        description="Calculates reflectivity using Shuey's approximation of the Zoeppritz equations.",
        params={
            "vp1": {
                "type": "number",
                "description": "P-wave velocity of the first medium in m/s."
            },
            "vs1": {
                "type": "number",
                "description": "S-wave velocity of the first medium in m/s."
            },
            "rho1": {
                "type": "number",
                "description": "Density of the first medium in g/cm³."
            },
            "vp2": {
                "type": "number",
                "description": "P-wave velocity of the second medium in m/s."
            },
            "vs2": {
                "type": "number",
                "description": "S-wave velocity of the second medium in m/s."
            },
            "rho2": {
                "type": "number",
                "description": "Density of the second medium in g/cm³."
            },
            "angles": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of incidence angles in degrees."
            },
        },
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"],
        defaults={},
        validator=None,  # TODO(Task 8): validate_avo
        auto_plot="plot_avo_reflectivity",
    ),
    ToolSpec(
        name="plot_avo_reflectivity",
        fn=plot_avo_reflectivity,
        description="Plots AVO reflectivity curves.",
        params={
            "angles": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of incidence angles in degrees."
            },
            "rc": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Array of reflection coefficients."
            },
        },
        required=["angles", "rc"],
        defaults={},
    ),
    ToolSpec(
        name="calculate_rock_properties",
        fn=calculate_rock_properties,
        description="Calculates Vp, Vs, density (rhob), Vp/Vs ratio, acoustic impedance, and shear impedance from porosity (phit) and clay volume (vclay) using empirical rock physics relationships. Returns all calculated values without plotting.",
        params={
            "phit": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Porosity values (fraction, 0-1)."
            },
            "vclay": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "Clay volume values (fraction, 0-1)."
            },
            "fluid_type": {
                "type": "string",
                "description": "Fluid type ('water', 'oil', or 'gas'). Default is 'water'."
            },
        },
        required=["phit", "vclay"],
        defaults={"fluid_type": "water"},
    ),
    ToolSpec(
        name="rock_physics_rag",
        fn=rock_physics_rag,
        description="Retrieves rock physics information using RAG (Retrieval-Augmented Generation).",
        params={
            "query": {
                "type": "string",
                "description": "The user's query about rock physics concepts."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of most relevant documents to retrieve (default: 3)."
            },
        },
        required=["query"],
        defaults={"top_k": 3},
    ),
    ToolSpec(
        name="knowledge_rag",
        fn=knowledge_rag,
        description="Retrieves information from the knowledge base using RAG (Retrieval-Augmented Generation) across all topics.",
        params={
            "query": {
                "type": "string",
                "description": "The user's query about any seismic or geophysics topic."
            },
            "domain": {
                "type": "string",
                "description": "Optional domain to restrict search (ricker, wedge, seismic_properties, rock_physics)."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of most relevant documents to retrieve (default: 3)."
            },
        },
        required=["query"],
        defaults={"domain": None, "top_k": 3},
    ),
]


def to_openai_schema(spec: ToolSpec) -> dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": {
            "type": "object",
            "properties": spec.params,
            "required": list(spec.required),
        },
    }


REGISTRY_BY_NAME = {s.name: s for s in REGISTRY}
TOOL_SCHEMAS = [to_openai_schema(s) for s in REGISTRY]
TOOL_FUNCTIONS = {s.name: s.fn for s in REGISTRY}
AUTO_PLOT = {s.name: s.auto_plot for s in REGISTRY if s.auto_plot}
