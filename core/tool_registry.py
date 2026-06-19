"""
Single source of truth for the seismic chatbot's LLM-facing tools.

Every tool is declared once as a ToolSpec. The OpenAI/DeepSeek schemas,
the name->function map, the auto-plot chaining map, and the validation
wiring are all DERIVED from REGISTRY — nothing else is hand-maintained.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools.ricker_tools import create_ricker_wavelet, create_ormsby_wavelet, plot_wavelet
from tools.wedge_tools import create_wedge_model, plot_wedge_model, analyze_wedge, wedge_avo_gather, plot_wedge_gather, analyze_wedge_gather
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity, avo_attributes, plot_avo_crossplot, extended_elastic_impedance, plot_extended_elastic_impedance
from tools.rock_physics_tools import calculate_rock_properties, rock_physics_rag, gassmann_substitution
from tools.rag_tools import knowledge_rag
from workflows.adapters import predict_elastic_layer
from tools.parameter_validation import validate_make_ricker, validate_wedge_model, validate_avo


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable
    description: str
    params: dict[str, dict]
    required: list[str]
    defaults: dict = field(default_factory=dict)
    validator: Optional[Callable] = None
    auto_plot: Optional[str] = None


_AVO_PARAMS: dict[str, dict] = {
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
}

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
        validator=validate_make_ricker,
        auto_plot="plot_ricker",
    ),
    ToolSpec(
        name="make_ormsby",
        fn=create_ormsby_wavelet,
        description="Creates an Ormsby (bandpass) wavelet from four corner frequencies f1<f2<f3<f4 in Hz.",
        params={
            "f1": {"type": "number", "description": "Low-cut corner frequency in Hz."},
            "f2": {"type": "number", "description": "Low-pass corner frequency in Hz."},
            "f3": {"type": "number", "description": "High-pass corner frequency in Hz."},
            "f4": {"type": "number", "description": "High-cut corner frequency in Hz."},
            "time_length": {"type": "number", "description": "Total length of the wavelet in milliseconds (default: 256 ms)."},
            "dt": {"type": "number", "description": "Time sampling interval in seconds (default: 0.001)."},
        },
        required=["f1", "f2", "f3", "f4"],
        defaults={"time_length": 256.0, "dt": 0.001},
        validator=None,
        auto_plot="plot_ricker",
    ),
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
        required=["wavelet", "time_array"],
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
        validator=validate_wedge_model,
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
    ToolSpec(
        name="analyze_wedge",
        fn=analyze_wedge,
        description="Analyzes a wedge model: returns tuning thickness, tuning amplitude, resolution limit, and the amplitude-vs-thickness curve.",
        params={
            "synthetic_data": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "2D array of synthetic seismic data."},
            "parameters": {"type": "object", "description": "Parameters returned by wedge_model."},
        },
        required=["synthetic_data", "parameters"],
        defaults={},
    ),
    ToolSpec(
        name="wedge_avo_gather",
        fn=wedge_avo_gather,
        description="Builds a wedge AVO angle gather: the synthetic wedge computed per incidence angle (Shuey), returned as a 3-D cube (time x thickness x angle).",
        params={
            "max_thickness": {"type": "number", "description": "Maximum thickness of the wedge layer in meters."},
            "v1": {"type": "number", "description": "P-wave velocity of the first layer in m/s."},
            "v2": {"type": "number", "description": "P-wave velocity of the second (wedge) layer in m/s."},
            "v3": {"type": "number", "description": "P-wave velocity of the third layer in m/s."},
            "rho1": {"type": "number", "description": "Density of the first layer in g/cm³."},
            "rho2": {"type": "number", "description": "Density of the second (wedge) layer in g/cm³."},
            "rho3": {"type": "number", "description": "Density of the third layer in g/cm³."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees (one synthetic panel per angle)."},
            "vs1": {"type": "number", "description": "S-wave velocity of layer 1 in m/s (optional, defaults to v1/2)."},
            "vs2": {"type": "number", "description": "S-wave velocity of layer 2 in m/s (optional, defaults to v2/2)."},
            "vs3": {"type": "number", "description": "S-wave velocity of layer 3 in m/s (optional, defaults to v3/2)."},
            "wavelet_freq": {"type": "number", "description": "Ricker wavelet frequency in Hz (default 30)."},
            "num_traces": {"type": "integer", "description": "Number of thickness traces (default 61)."},
        },
        required=["max_thickness", "v1", "v2", "v3", "rho1", "rho2", "rho3", "angles"],
        defaults={"wavelet_freq": 30.0, "num_traces": 61},
        validator=validate_wedge_model,
        auto_plot="plot_wedge_gather",
    ),
    ToolSpec(
        name="plot_wedge_gather",
        fn=plot_wedge_gather,
        description="Plots a wedge AVO gather: amplitude-vs-thickness per angle and amplitude-vs-angle (AVO) at maximum thickness.",
        params={
            "gather": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "description": "3-D gather cube (time x thickness x angle) from wedge_avo_gather."},
            "parameters": {"type": "object", "description": "Parameters returned by wedge_avo_gather."},
        },
        required=["gather", "parameters"],
        defaults={},
    ),
    ToolSpec(
        name="analyze_wedge_gather",
        fn=analyze_wedge_gather,
        description="Analyzes a wedge AVO gather: per-angle tuning thickness/amplitude and the AVO curve at maximum thickness.",
        params={
            "gather": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "description": "3-D gather cube (time x thickness x angle) from wedge_avo_gather."},
            "parameters": {"type": "object", "description": "Parameters returned by wedge_avo_gather."},
        },
        required=["gather", "parameters"],
        defaults={},
    ),
    ToolSpec(
        name="zoeppritz_reflectivity",
        fn=zoeppritz_reflectivity,
        description="Calculates reflectivity using the Zoeppritz equations for elastic wave reflection.",
        params=_AVO_PARAMS,
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"],
        defaults={},
        validator=validate_avo,
        auto_plot="plot_avo_reflectivity",
    ),
    ToolSpec(
        name="shuey_reflectivity",
        fn=shuey_reflectivity,
        description="Calculates reflectivity using Shuey's approximation of the Zoeppritz equations.",
        params=_AVO_PARAMS,
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"],
        defaults={},
        validator=validate_avo,
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
        name="avo_attributes",
        fn=avo_attributes,
        description="Computes AVO interpretation attributes for an interface: intercept (A), gradient (B), and AVO class (I-IV) from the two media's Vp, Vs, and density. Auto-plots the intercept-gradient crossplot.",
        params={
            "vp1": {"type": "number", "description": "P-wave velocity of the upper medium in m/s."},
            "vs1": {"type": "number", "description": "S-wave velocity of the upper medium in m/s."},
            "rho1": {"type": "number", "description": "Density of the upper medium in g/cm³."},
            "vp2": {"type": "number", "description": "P-wave velocity of the lower medium in m/s."},
            "vs2": {"type": "number", "description": "S-wave velocity of the lower medium in m/s."},
            "rho2": {"type": "number", "description": "Density of the lower medium in g/cm³."},
        },
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2"],
        defaults={},
        validator=None,
        auto_plot="plot_avo_crossplot",
    ),
    ToolSpec(
        name="plot_avo_crossplot",
        fn=plot_avo_crossplot,
        description="Plots the AVO intercept-gradient (A-B) crossplot with shaded class regions and the marked point.",
        params={
            "intercept": {"type": "number", "description": "AVO intercept A."},
            "gradient": {"type": "number", "description": "AVO gradient B."},
            "avo_class": {"type": "string", "description": "Optional AVO class label to annotate the point."},
        },
        required=["intercept", "gradient"],
        defaults={},
    ),
    ToolSpec(
        name="extended_elastic_impedance",
        fn=extended_elastic_impedance,
        description="Computes Extended Elastic Impedance EEI(χ) (Whitcombe 2002) for a layer across rotation angles χ in degrees. At χ=0 it equals the acoustic impedance Vp·ρ. Optionally Whitcombe-normalized when reference constants vp0/vs0/rho0 are all supplied. Auto-plots EEI vs χ.",
        params={
            "vp": {"type": "number", "description": "P-wave velocity of the layer in m/s."},
            "vs": {"type": "number", "description": "S-wave velocity of the layer in m/s."},
            "rho": {"type": "number", "description": "Density of the layer in g/cm³."},
            "chi": {"type": "array", "items": {"type": "number"}, "description": "Rotation angles χ in degrees (|χ| ≤ 90)."},
            "vp0": {"type": "number", "description": "Optional reference P-wave velocity (m/s) for Whitcombe normalization; supply with vs0 and rho0."},
            "vs0": {"type": "number", "description": "Optional reference S-wave velocity (m/s) for Whitcombe normalization; supply with vp0 and rho0."},
            "rho0": {"type": "number", "description": "Optional reference density (g/cm³) for Whitcombe normalization; supply with vp0 and vs0."},
            "k": {"type": "number", "description": "Optional background (Vs/Vp)² constant; defaults to (vs/vp)² of the layer."},
        },
        required=["vp", "vs", "rho", "chi"],
        defaults={},
        validator=None,
        auto_plot="plot_extended_elastic_impedance",
    ),
    ToolSpec(
        name="plot_extended_elastic_impedance",
        fn=plot_extended_elastic_impedance,
        description="Plots Extended Elastic Impedance EEI vs rotation angle χ.",
        params={
            "chi": {"type": "array", "items": {"type": "number"}, "description": "Rotation angles χ in degrees."},
            "eei": {"type": "array", "items": {"type": "number"}, "description": "EEI values, one per χ."},
        },
        required=["chi", "eei"],
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
        name="gassmann_substitution",
        fn=gassmann_substitution,
        description="Gassmann fluid substitution: from in-situ Vp, Vs, density and porosity, compute the elastic properties (Vp, Vs, density) after swapping the pore fluid (e.g. brine to gas). Shear modulus is held fluid-independent.",
        params={
            "vp": {"type": "number", "description": "In-situ P-wave velocity in m/s."},
            "vs": {"type": "number", "description": "In-situ S-wave velocity in m/s."},
            "rho": {"type": "number", "description": "In-situ bulk density in g/cm³."},
            "phi": {"type": "number", "description": "Porosity (fraction, 0-1)."},
            "fluid_in": {"type": "string", "description": "In-situ pore fluid: 'water'/'brine', 'oil', or 'gas'."},
            "fluid_out": {"type": "string", "description": "Target pore fluid to substitute in: 'water'/'brine', 'oil', or 'gas'."},
            "k_mineral": {"type": "number", "description": "Mineral (grain) bulk modulus in GPa (default 37, quartz)."},
            "k_fl_in": {"type": "number", "description": "Optional in-situ fluid bulk-modulus override in GPa."},
            "rho_fl_in": {"type": "number", "description": "Optional in-situ fluid density override in g/cm³."},
            "k_fl_out": {"type": "number", "description": "Optional target fluid bulk-modulus override in GPa."},
            "rho_fl_out": {"type": "number", "description": "Optional target fluid density override in g/cm³."},
        },
        required=["vp", "vs", "rho", "phi", "fluid_in", "fluid_out"],
        defaults={"k_mineral": 37.0},
        validator=None,
        auto_plot=None,
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
    ToolSpec(
        name="predict_elastic_layer",
        fn=predict_elastic_layer,
        description="Predicts representative elastic properties (Vp, Vs, density, Vp/Vs ratio) of a single rock layer from porosity (phit) and clay volume (vclay) using the Han et al. (1986) rock-physics model. Reduces a log of samples to one representative scalar layer. Returns a dict; no plot.",
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity values (fraction, 0-1); a single value or a log array."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay volume values (fraction, 0-1)."},
            "fluid": {"type": "string", "description": "Pore fluid: 'water'/'brine', 'oil', or 'gas' (default 'water')."},
            "reduce": {"type": "string", "description": "How to reduce a log array to one scalar layer: 'mean' or 'median' (default 'mean')."},
        },
        required=["phit", "vclay"],
        defaults={"fluid": "water", "reduce": "mean"},
        validator=None,
        auto_plot=None,
    ),
]


def to_openai_schema(spec: ToolSpec) -> dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": {
            "type": "object",
            "properties": dict(spec.params),
            "required": list(spec.required),
        },
    }


REGISTRY_BY_NAME = {s.name: s for s in REGISTRY}
TOOL_SCHEMAS = [to_openai_schema(s) for s in REGISTRY]
TOOL_FUNCTIONS = {s.name: s.fn for s in REGISTRY}
AUTO_PLOT = {s.name: s.auto_plot for s in REGISTRY if s.auto_plot}
