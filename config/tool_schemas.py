"""
Tool schemas for the seismic chatbot following the tool use pattern.
These schemas define the tools available to the LLM with proper JSON structure.
"""

TOOL_SCHEMAS = [
    {
        "name": "knowledge_rag",
        "description": "Retrieves information from the knowledge base using RAG (Retrieval-Augmented Generation) across all topics.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "rock_physics_rag",
        "description": "Retrieves rock physics information using RAG (Retrieval-Augmented Generation).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query about rock physics concepts."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant documents to retrieve (default: 3)."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "make_ricker",
        "description": "Creates a Ricker wavelet with specified frequency and time parameters.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["frequency"]
        }
    },
    {
        "name": "plot_ricker",
        "description": "Plots a Ricker wavelet with time domain and frequency domain analysis.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["wavelet"]
        }
    },
    {
        "name": "wedge_model",
        "description": "Creates a wedge model for seismic analysis with variable thickness layers.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["max_thickness", "v1", "v2", "v3", "rho1", "rho2", "rho3"]
        }
    },
    {
        "name": "plot_wedge_model",
        "description": "Plots a wedge model showing seismic response vs thickness.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["synthetic_data", "parameters"]
        }
    },
    {
        "name": "zoeppritz_reflectivity",
        "description": "Calculates reflectivity using the Zoeppritz equations for elastic wave reflection.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"]
        }
    },
    {
        "name": "shuey_reflectivity",
        "description": "Calculates reflectivity using Shuey's approximation of the Zoeppritz equations.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"]
        }
    },
    {
        "name": "plot_avo_reflectivity",
        "description": "Plots AVO reflectivity curves.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["angles", "rc"]
        }
    },
    {
        "name": "calculate_rock_properties",
        "description": "Calculates Vp, Vs, and density (rhob) from porosity (phit) and clay volume (vclay) using empirical rock physics relationships.",
        "parameters": {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["phit", "vclay"]
        }
    },
    {
        "name": "plot_rock_properties",
        "description": "Plots calculated rock properties (Vp, Vs, density) as a function of porosity and clay volume.",
        "parameters": {
            "type": "object",
            "properties": {
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
                "vp": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "P-wave velocity values (m/s)."
                },
                "vs": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "S-wave velocity values (m/s)."
                },
                "rhob": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "Bulk density values (g/cc)."
                }
            },
            "required": ["phit", "vclay", "vp", "vs", "rhob"]
        }
    }
]

# Tool function mapping for execution
TOOL_FUNCTIONS = {
    "make_ricker": "tools.ricker_tools.create_ricker_wavelet",
    "plot_ricker": "tools.ricker_tools.plot_wavelet",
    "wedge_model": "tools.wedge_tools.create_wedge_model",
    "plot_wedge_model": "tools.wedge_tools.plot_wedge_model",
    "zoeppritz_reflectivity": "tools.avo_tools.zoeppritz_reflectivity",
    "shuey_reflectivity": "tools.avo_tools.shuey_reflectivity",
    "plot_avo_reflectivity": "tools.avo_tools.plot_avo_reflectivity",
    "calculate_rock_properties": "tools.rock_physics_tools.calculate_rock_properties",
    "plot_rock_properties": "tools.rock_physics_tools.plot_rock_properties",
    "rock_physics_rag": "tools.rock_physics_tools.rock_physics_rag",
    "knowledge_rag": "tools.rag_tools.knowledge_rag"
}