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
            "title": "Create 50 Hz Ricker",
            "prompt": "Create a 50 Hz Ricker wavelet with 0.001s sampling",
            "description": "Generate a higher frequency wavelet with specific sampling"
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
            "title": "Gas sand AVO wedge",
            "prompt": "Create a wedge model for gas sand using Shuey approximation with incident angle=20, v1=2200, v2=1800, v3=2400, vs1=1100, vs2=900, vs3=1200, rho1=2.2, rho2=1.9, rho3=2.3",
            "description": "Model gas sand with angle-dependent reflectivity"
        },
        {
            "title": "Wedge model with incident angle 30 degrees",
            "prompt": "Make a wedge model with incident angle 30 degrees, v1=2000, v2=2500, v3=3000, rho1=2.1, rho2=2.3, rho3=2.5",
            "description": "Create wedge model with specific incident angle"
        },
        {
            "title": "Oil sand wedge at 25 degrees",
            "prompt": "Create an oil sand wedge model with angle 25 degrees, velocities [2200, 2000, 2400], densities [2.2, 2.0, 2.3], max_thickness=120",
            "description": "Model oil sand with angle-dependent reflectivity"
        },
        {
            "title": "High angle wedge model",
            "prompt": "Generate a wedge model with incident angle=45 degrees for v1=1800, v2=2200, v3=2600, rho1=1.9, rho2=2.2, rho3=2.4",
            "description": "Model with high incident angle for AVO analysis"
        },
        {
            "title": "Multiple angle wedge comparison",
            "prompt": "Create wedge models with incident angles [0, 15, 30] degrees for v1=2000, v2=2500, v3=3000, rho1=2.1, rho2=2.3, rho3=2.5",
            "description": "Compare wedge models at different incident angles"
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
            "title": "Plot AVO curves",
            "prompt": "Plot AVO reflectivity for angles=[0,5,10,15,20,25,30], rc=[0.1,0.08,0.05,0.02,-0.01,-0.03,-0.05]",
            "description": "Visualize AVO curves"
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
            "title": "Oil sand properties",
            "prompt": "Calculate Vp, Vs, and density for porosity=0.18, clay_volume=0.25, fluid_type='oil'",
            "description": "Model oil sand elastic properties"
        },
        {
            "title": "Plot rock properties",
            "prompt": "Calculate and plot rock properties for porosity range [0.1, 0.3] and clay volume [0.1, 0.5] with water saturation",
            "description": "Visualize how rock properties vary with porosity and clay content"
        },
        {
            "title": "Shale properties",
            "prompt": "Calculate elastic properties for high clay content: porosity=0.12, clay_volume=0.8, fluid_type='water'",
            "description": "Model shale rock properties"
        },
        {
            "title": "Sandstone properties",
            "prompt": "Calculate Vp, Vs, and density for clean sandstone: porosity=0.22, clay_volume=0.05, fluid_type='water'",
            "description": "Model clean sandstone properties"
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
    
    "Advanced Topics": [
        {
            "title": "Tuning analysis",
            "prompt": "Create a wedge model and analyze the tuning thickness for a 30 Hz wavelet",
            "description": "Analyze tuning effects in seismic data"
        },
        {
            "title": "Resolution limits",
            "prompt": "What are the resolution limits for a 25 Hz wavelet and how do they affect interpretation?",
            "description": "Understand seismic resolution limitations"
        },
        {
            "title": "AVO classification",
            "prompt": "How do you classify AVO responses and what do different classes indicate?",
            "description": "Learn about AVO classification schemes"
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