# Tools for rock physics calculations
import numpy as np
from typing import Dict, List, Any, Optional, Union
from knowledge.vector_db import VectorDatabase
from knowledge.topics.rock_physics import ROCK_PHYSICS_KNOWLEDGE

def _format_value(value: Union[float, np.ndarray], precision: int = 3) -> str:
    """
    Format a value (scalar or array) for printing.
    
    Args:
        value: Scalar float or numpy array
        precision: Number of decimal places
        
    Returns:
        Formatted string representation
    """
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=precision)
    else:
        return f"{value:.{precision}f}"

def calculate_rock_properties(phit, vclay, fluid_type='water', print_results=True):
    """
    Calculate Vp, Vs, density (rhob), impedance, and Vp/Vs ratio from porosity (phit) and clay volume (vclay)
    using empirical rock physics relationships.
    
    Args:
        phit: float or array-like, porosity (fraction, 0-1)
        vclay: float or array-like, clay volume (fraction, 0-1)
        fluid_type: str, fluid type ('water', 'oil', or 'gas')
        print_results: bool, whether to print the calculated values
        
    Returns:
        tuple: (vp, vs, rhob, vp_vs_ratio, ai, si) - P-wave velocity (m/s), S-wave velocity (m/s), 
               bulk density (g/cc), Vp/Vs ratio, acoustic impedance (×10⁶ kg/m²·s), 
               and shear impedance (×10⁶ kg/m²·s)
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
    
    # Calculate Vp/Vs ratio
    vp_vs_ratio = vp / vs
    
    # Calculate impedance values
    # Convert density from g/cc to kg/m³ for impedance calculation
    rho_kg_m3 = rhob * 1000
    ai = rho_kg_m3 * vp / 1e6  # Acoustic impedance in ×10⁶ kg/m²·s
    si = rho_kg_m3 * vs / 1e6  # Shear impedance in ×10⁶ kg/m²·s
    
    # Print results if requested
    if print_results:
        print(f"\n=== Rock Properties Calculation Results ===")
        print(f"Input Parameters:")
        print(f"  Porosity (phit): {phit}")
        print(f"  Clay Volume (vclay): {vclay}")
        print(f"  Fluid Type: {fluid_type}")
        print(f"\nCalculated Properties:")
        print(f"  P-wave Velocity (Vp): {_format_value(vp, 0)} m/s")
        print(f"  S-wave Velocity (Vs): {_format_value(vs, 0)} m/s")
        print(f"  Bulk Density (rhob): {_format_value(rhob, 3)} g/cc")
        print(f"  Vp/Vs Ratio: {_format_value(vp_vs_ratio, 2)}")
        print(f"  Acoustic Impedance (AI): {_format_value(ai, 2)} × 10⁶ kg/m²·s")
        print(f"  Shear Impedance (SI): {_format_value(si, 2)} × 10⁶ kg/m²·s")
        print(f"  Matrix Density: {_format_value(rho_matrix, 3)} g/cc")
        print(f"  Fluid Density: {_format_value(rho_fluid, 3)} g/cc")
        print("=" * 40)
    
    return vp, vs, rhob, vp_vs_ratio, ai, si

def calculate_elastic_moduli(vp, vs, rhob):
    """
    Calculate elastic moduli from velocities and density.
    
    Args:
        vp: P-wave velocity (m/s)
        vs: S-wave velocity (m/s)
        rhob: bulk density (g/cc)
        
    Returns:
        tuple: (K, G, E, nu) - bulk modulus, shear modulus, Young's modulus, and Poisson's ratio
    """
    # Convert density from g/cc to kg/m³
    rho_kg_m3 = rhob * 1000
    
    # Calculate elastic moduli
    K = rho_kg_m3 * (vp**2 - 4/3 * vs**2)  # Bulk modulus (Pa)
    G = rho_kg_m3 * vs**2  # Shear modulus (Pa)
    E = 9 * K * G / (3 * K + G)  # Young's modulus (Pa)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))  # Poisson's ratio
    
    # Print results
    print(f"\n=== Elastic Moduli Calculation ===")
    print(f"Bulk Modulus (K): {_format_value(K/1e9, 2)} GPa")
    print(f"Shear Modulus (G): {_format_value(G/1e9, 2)} GPa")
    print(f"Young's Modulus (E): {_format_value(E/1e9, 2)} GPa")
    print(f"Poisson's Ratio (ν): {_format_value(nu, 3)}")
    print("=" * 35)
    
    return K, G, E, nu

def calculate_impedance(vp, vs, rhob):
    """
    Calculate acoustic and shear impedance.
    
    Args:
        vp: P-wave velocity (m/s)
        vs: S-wave velocity (m/s)
        rhob: bulk density (g/cc)
        
    Returns:
        tuple: (AI, SI) - acoustic impedance and shear impedance
    """
    # Convert density from g/cc to kg/m³
    rho_kg_m3 = rhob * 1000
    
    # Calculate impedances
    AI = rho_kg_m3 * vp  # Acoustic impedance (kg/m²·s)
    SI = rho_kg_m3 * vs  # Shear impedance (kg/m²·s)
    
    # Print results
    print(f"\n=== Impedance Calculation ===")
    print(f"Acoustic Impedance (AI): {_format_value(AI/1e6, 2)} × 10⁶ kg/m²·s")
    print(f"Shear Impedance (SI): {_format_value(SI/1e6, 2)} × 10⁶ kg/m²·s")
    print("=" * 30)
    
    return AI, SI


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