"""
Example prompts configuration for the Seismic ChatBot.
These prompts can be copied directly by users to get started quickly.
"""

EXAMPLE_PROMPTS = {
    "Educational Questions": [
        {
            "title": "What is a Ricker wavelet?",
            "prompt": "What is a Ricker wavelet and why is it important in seismic analysis?",
            "description": "Learn about the fundamental wavelet used in seismic analysis"
        },
        {
            "title": "How does frequency affect resolution?",
            "prompt": "How does frequency affect seismic resolution and what are the trade-offs?",
            "description": "Understand the relationship between frequency and resolution"
        },
        {
            "title": "Explain tuning thickness",
            "prompt": "What is tuning thickness and how does it affect seismic interpretation?",
            "description": "Learn about tuning effects in seismic data"
        },
        {
            "title": "What is AVO analysis?",
            "prompt": "What is AVO (Amplitude Versus Offset) analysis and when is it used?",
            "description": "Understand AVO analysis principles"
        },
        {
            "title": "Zoeppritz vs Shuey",
            "prompt": "What's the difference between Zoeppritz and Shuey equations for AVO analysis?",
            "description": "Compare different AVO calculation methods"
        }
    ],
    
    "Wavelet Tools": [
        {
            "title": "Create 30 Hz Ricker",
            "prompt": "Create a 30 Hz Ricker wavelet",
            "description": "Generate a standard Ricker wavelet"
        },
        {
            "title": "Create Ormsby wavelet",
            "prompt": "Create an Ormsby bandpass wavelet with corner frequencies f1=5, f2=10, f3=40, f4=60 Hz",
            "description": "Generate a bandpass Ormsby wavelet from four corner frequencies"
        },
        {
            "title": "Plot wavelet spectrum",
            "prompt": "Plot the spectrum of a 40 Hz Ricker wavelet",
            "description": "Visualize the frequency content of a wavelet"
        }
    ],
    
    "Wedge Modeling": [
        {
            "title": "Simple wedge model",
            "prompt": "Make a wedge model with 100m thickness, v1=2000, v2=2500, v3=3000, rho1=2.0, rho2=2.2, rho3=2.4",
            "description": "Create a basic wedge model with specified parameters"
        },
        {
            "title": "Gas sand wedge",
            "prompt": "Create a wedge model for a gas sand with v1=2000, v2=1800, v3=2200, rho1=2.1, rho2=1.8, rho3=2.2, max_thickness=150",
            "description": "Model a gas sand scenario"
        },
        {
            "title": "Oil sand wedge",
            "prompt": "Make a wedge model for oil sand with velocities [2200, 2000, 2400] and densities [2.2, 2.0, 2.3], 120m thickness",
            "description": "Model an oil sand scenario"
        },
        {
            "title": "AVO wedge model at 15 degrees",
            "prompt": "Create a wedge model using Shuey approximation with incident angle=15, v1=2000, v2=2500, v3=3000, vs1=1000, vs2=1200, vs3=1500, rho1=2.1, rho2=2.3, rho3=2.5",
            "description": "Model angle-dependent reflectivity at 15 degrees"
        },
        {
            "title": "Wedge AVO angle gather",
            "prompt": "Build a wedge AVO angle gather over angles [0, 10, 20, 30, 40] with max_thickness=60, v1=2000, v2=2500, v3=3000, vs1=1000, vs2=1200, vs3=1500, rho1=2.1, rho2=2.3, rho3=2.5",
            "description": "Model a true per-angle wedge gather (one synthetic panel per incidence angle)"
        }
    ],
    
    "AVO Analysis": [
        {
            "title": "Zoeppritz reflectivity",
            "prompt": "Calculate Zoeppritz reflectivity for vp1=2000, vs1=800, rho1=2.0, vp2=2500, vs2=1000, rho2=2.2, angles=[0,10,20,30]",
            "description": "Calculate reflectivity using Zoeppritz equations"
        },
        {
            "title": "Shuey reflectivity",
            "prompt": "Calculate Shuey reflectivity for vp1=1800, vs1=600, rho1=1.8, vp2=2200, vs2=800, rho2=2.1, angles=[0,5,10,15,20,25,30]",
            "description": "Calculate reflectivity using Shuey's approximation"
        },
        {
            "title": "AVO attributes & class",
            "prompt": "Compute the AVO intercept, gradient, and class for an interface with vp1=2500, vs1=1200, rho1=2.3, vp2=2200, vs2=1300, rho2=2.0",
            "description": "Get the intercept (A), gradient (B), and AVO class with the A-B crossplot"
        },
        {
            "title": "Gas sand AVO",
            "prompt": "Calculate Zoeppritz reflectivity for gas sand: vp1=2200, vs1=800, rho1=2.1, vp2=1800, vs2=600, rho2=1.8, angles=[0,5,10,15,20,25,30]",
            "description": "Model gas sand AVO response"
        }
    ],
    
    "Rock Physics": [
        {
            "title": "Calculate rock properties",
            "prompt": "Calculate Vp, Vs, and density for porosity=0.25, clay_volume=0.3, fluid_type='water'",
            "description": "Calculate elastic properties from porosity and clay content"
        },
        {
            "title": "Gas sand properties",
            "prompt": "Calculate rock properties for porosity=0.20, clay_volume=0.15, fluid_type='gas'",
            "description": "Model gas sand elastic properties"
        },
        {
            "title": "Plot rock properties",
            "prompt": "Calculate and plot rock properties for porosity range [0.1, 0.3] and clay volume [0.1, 0.5] with water saturation",
            "description": "Visualize how rock properties vary with porosity and clay content"
        },
        {
            "title": "Gassmann fluid substitution",
            "prompt": "Run Gassmann fluid substitution from brine to gas for vp=2800, vs=1500, rho=2.2, porosity=0.25",
            "description": "Substitute the pore fluid and get the new Vp, Vs, and density"
        },
        {
            "title": "Saturation sweep",
            "prompt": "Calculate Vp, Vs and Vp/Vs across water saturation [0, 0.2, 0.4, 0.6, 0.8, 1.0] for porosity=0.25, clay_volume=0.15 with a gas hydrocarbon",
            "description": "Model elastic properties along a fluid-saturation line"
        },
        {
            "title": "Predict layer from logs",
            "prompt": "Predict the representative Vp, Vs and density of a sand from porosity log [0.22, 0.25, 0.2, 0.27] and clay log [0.1, 0.15, 0.08, 0.12], gas-filled",
            "description": "Reduce porosity/clay logs to one representative elastic layer (Han 1986)"
        },
        {
            "title": "Rock physics RAG query",
            "prompt": "What are the key factors that affect P-wave velocity in rocks?",
            "description": "Use RAG to get information about velocity controls"
        },
        {
            "title": "Fluid substitution effects",
            "prompt": "How do different fluids affect seismic velocities and what are the implications for AVO analysis?",
            "description": "Learn about fluid effects on rock properties"
        },
        {
            "title": "Elastic modulus relationships",
            "prompt": "What are the relationships between bulk modulus, shear modulus, and seismic velocities?",
            "description": "Understand elastic property relationships"
        }
    ],
    
    "Workflows & Advanced Analysis": [
        {
            "title": "Petro-to-AVO feasibility",
            "prompt": "Run a petro-to-AVO feasibility for a gas sand with porosity 0.25 and clay volume 0.15 below a shale with porosity 0.1 and clay volume 0.6, over angles [0, 10, 20, 30, 40]",
            "description": "End-to-end workflow: predict elastic properties, build the interface, and model the AVO curve + attributes"
        },
        {
            "title": "Fluid scenario comparison",
            "prompt": "Compare the AVO response of a sand (porosity 0.25, clay 0.15) under a shale (porosity 0.1, clay 0.6) for brine vs gas across angles [0, 10, 20, 30, 40]",
            "description": "Gassmann fluid-substitution scenarios overlaid for DHI / fluid feasibility"
        },
        {
            "title": "Wedge tuning from petrophysics",
            "prompt": "Analyze wedge tuning for a sand (porosity 0.25, clay 0.15) encased in shale (porosity 0.1, clay 0.6) with 60 m max thickness and a 30 Hz wavelet",
            "description": "Predict the layers, build the wedge, and find the tuning thickness and resolution limit"
        },
        {
            "title": "EEI optimal rotation angle",
            "prompt": "Find the EEI rotation angle that best correlates with clay volume for porosity log [0.2, 0.25, 0.18, 0.3] and clay log [0.4, 0.2, 0.6, 0.1]",
            "description": "Extended Elastic Impedance: scan chi for the best lithology/porosity/Sw discriminator"
        },
        {
            "title": "Parameter sweep / sensitivity",
            "prompt": "Sweep the petro_to_avo gradient over sand porosity [0.1, 0.2, 0.3] and fluid [brine, gas], holding the shale and angles [0, 10, 20, 30] fixed",
            "description": "Run a recipe across a parameter grid and summarize how an output responds"
        }
    ]
}

# Quick access to all prompts for search functionality
ALL_PROMPTS = []
for category, prompts in EXAMPLE_PROMPTS.items():
    for prompt in prompts:
        ALL_PROMPTS.append({
            "category": category,
            "title": prompt["title"],
            "prompt": prompt["prompt"],
            "description": prompt["description"]
        })

def get_prompts_by_category(category: str) -> list:
    """Get all prompts for a specific category."""
    return EXAMPLE_PROMPTS.get(category, [])

def search_prompts(query: str) -> list:
    """Search prompts by title, description, or prompt text."""
    query = query.lower()
    results = []
    
    for prompt in ALL_PROMPTS:
        if (query in prompt["title"].lower() or 
            query in prompt["description"].lower() or 
            query in prompt["prompt"].lower()):
            results.append(prompt)
    
    return results

def get_random_prompts(count: int = 3) -> list:
    """Get random prompts for suggestions."""
    import random
    return random.sample(ALL_PROMPTS, min(count, len(ALL_PROMPTS)))