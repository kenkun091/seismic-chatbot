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