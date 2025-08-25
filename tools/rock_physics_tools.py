# Tools for rock physics calculations
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from typing import Dict, List, Any, Optional
from knowledge.vector_db import VectorDatabase
from knowledge.topics.rock_physics import ROCK_PHYSICS_KNOWLEDGE

def calculate_rock_properties(phit, vclay, fluid_type='water'):
    """
    Calculate Vp, Vs, and density (rhob) from porosity (phit) and clay volume (vclay)
    using empirical rock physics relationships.
    
    Args:
        phit: float or array-like, porosity (fraction, 0-1)
        vclay: float or array-like, clay volume (fraction, 0-1)
        fluid_type: str, fluid type ('water', 'oil', or 'gas')
        
    Returns:
        tuple: (vp, vs, rhob) - P-wave velocity (m/s), S-wave velocity (m/s), and bulk density (g/cc)
    """
    # Convert inputs to numpy arrays if they aren't already
    phit = np.asarray(phit)
    vclay = np.asarray(vclay)
    
    # Ensure values are within valid ranges
    phit = np.clip(phit, 0.01, 0.4)
    vclay = np.clip(vclay, 0.05, 0.95)
    
    # Calculate matrix properties based on clay content
    # Linear interpolation between sandstone and shale end members
    # Sandstone properties (vclay = 0)
    vp_sand_matrix = 5500  # m/s
    vs_sand_matrix = 3400  # m/s
    rho_sand_matrix = 2.65  # g/cc
    
    # Shale properties (vclay = 1)
    vp_shale_matrix = 3800  # m/s
    vs_shale_matrix = 2000  # m/s
    rho_shale_matrix = 2.55  # g/cc
    
    # Interpolate matrix properties based on clay content
    vp_matrix = vp_sand_matrix * (1 - vclay) + vp_shale_matrix * vclay
    vs_matrix = vs_sand_matrix * (1 - vclay) + vs_shale_matrix * vclay
    rho_matrix = rho_sand_matrix * (1 - vclay) + rho_shale_matrix * vclay
    
    # Fluid properties
    if fluid_type.lower() == 'water':
        rho_fluid = 1.0  # g/cc
        k_fluid = 2.2e9  # Pa, bulk modulus of water
    elif fluid_type.lower() == 'oil':
        rho_fluid = 0.8  # g/cc
        k_fluid = 1.5e9  # Pa, bulk modulus of oil
    elif fluid_type.lower() == 'gas':
        rho_fluid = 0.2  # g/cc
        k_fluid = 0.1e9  # Pa, bulk modulus of gas
    else:
        raise ValueError(f"Unknown fluid type: {fluid_type}. Use 'water', 'oil', or 'gas'.")
    
    # Calculate bulk density using simple mixing law
    rhob = rho_matrix * (1 - phit) + rho_fluid * phit
    
    # Calculate velocities using modified Wyllie time-average equation with clay effect
    # Apply Raymer-Hunt-Gardner modifications for better accuracy
    # Vp calculation
    vp_factor = 1.0 - 0.5 * vclay  # Clay effect factor
    vp = vp_matrix * (1 - phit)**2 * vp_factor
    
    # Vs calculation (using Vp/Vs ratio that varies with clay content)
    vp_vs_ratio = 1.5 + 0.5 * vclay  # Increases with clay content
    vs = vp / vp_vs_ratio
    
    # Apply fluid effects (simplified Gassmann)
    # Reduce velocities for gas-filled porosity
    if fluid_type.lower() == 'gas':
        vp = vp * (1.0 - 0.3 * phit)  # Stronger effect on Vp
        vs = vs * (1.0 - 0.1 * phit)  # Weaker effect on Vs
    
    return vp, vs, rhob

def plot_rock_properties(phit, vclay, vp, vs, rhob, output_path=None):
    """
    Plot calculated rock properties as a function of porosity and clay volume.
    
    Args:
        phit: array-like, porosity values
        vclay: array-like, clay volume values
        vp: array-like, P-wave velocity values
        vs: array-like, S-wave velocity values
        rhob: array-like, bulk density values
        output_path: Optional path to save the plot. If None, creates a temporary file.
        
    Returns:
        str: Path to the saved plot file
    """
    import tempfile
    import os
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot Vp
    sc1 = axs[0].scatter(phit, vclay, c=vp, cmap='viridis', s=50)
    axs[0].set_xlabel('Porosity (fraction)')
    axs[0].set_ylabel('Clay Volume (fraction)')
    axs[0].set_title('P-wave Velocity (m/s)')
    plt.colorbar(sc1, ax=axs[0])
    
    # Plot Vs
    sc2 = axs[1].scatter(phit, vclay, c=vs, cmap='plasma', s=50)
    axs[1].set_xlabel('Porosity (fraction)')
    axs[1].set_ylabel('Clay Volume (fraction)')
    axs[1].set_title('S-wave Velocity (m/s)')
    plt.colorbar(sc2, ax=axs[1])
    
    # Plot density
    sc3 = axs[2].scatter(phit, vclay, c=rhob, cmap='cividis', s=50)
    axs[2].set_xlabel('Porosity (fraction)')
    axs[2].set_ylabel('Clay Volume (fraction)')
    axs[2].set_title('Bulk Density (g/cc)')
    plt.colorbar(sc3, ax=axs[2])
    
    plt.tight_layout()
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


# Initialize the vector database with rock physics knowledge
_rock_physics_db = None

def _get_rock_physics_db():
    """Get or initialize the rock physics vector database."""
    global _rock_physics_db
    
    if _rock_physics_db is None:
        _rock_physics_db = VectorDatabase()
        
        # Add rock physics knowledge to the database
        for topic, content in ROCK_PHYSICS_KNOWLEDGE.items():
            _rock_physics_db.add_document(
                text=content,
                metadata={'topic': topic}
            )
    
    return _rock_physics_db


def rock_physics_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve rock physics information using RAG (Retrieval-Augmented Generation).
    
    Args:
        query: The user's query about rock physics
        top_k: Number of most relevant documents to retrieve
        
    Returns:
        Dict containing retrieved information and metadata
    """
    # Get the vector database
    db = _get_rock_physics_db()
    
    # Search for relevant documents
    results = db.search(query, top_k=top_k)
    
    # Format the response
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append({
            'content': result['document'],
            'topic': result['metadata'].get('topic', 'unknown'),
            'relevance_score': float(result['score'])
        })
    
    # Create the response
    response = {
        'query': query,
        'results': formatted_results,
        'total_results': len(formatted_results)
    }
    
    return response