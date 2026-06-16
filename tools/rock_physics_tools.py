# Tools for rock physics calculations
import numpy as np
import warnings
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

# --- Reference moduli/densities (Mavko, Mukerji & Dvorkin, Rock Physics Handbook) ---
_K_QUARTZ = 37.0e9   # Pa, mineral bulk modulus of quartz
_K_CLAY = 21.0e9     # Pa, mineral bulk modulus of clay (ill-defined; central value)
_RHO_QUARTZ = 2.65   # g/cc
_RHO_CLAY = 2.58     # g/cc

# Fluid bulk modulus (Pa) and density (g/cc), Batzle-Wang typical reservoir values.
_FLUIDS = {
    'water': (2.2e9, 1.0),
    'brine': (2.2e9, 1.0),
    'oil':   (1.0e9, 0.8),
    'gas':   (0.05e9, 0.2),
}


def gassmann_sat(K_dry, K0, K_fl, phi):
    """Gassmann forward: saturated bulk modulus from dry frame, mineral, fluid, porosity.

    K_sat = K_dry + (1 - K_dry/K0)^2 / (phi/K_fl + (1-phi)/K0 - K_dry/K0^2)
    (Mavko et al., Rock Physics Handbook.)
    """
    return K_dry + (1.0 - K_dry / K0) ** 2 / (phi / K_fl + (1.0 - phi) / K0 - K_dry / K0 ** 2)


def gassmann_dry(K_sat, K0, K_fl, phi):
    """Gassmann inverse: recover the dry-frame modulus from a saturated modulus
    (Kumar, 2006). Exact inverse of ``gassmann_sat``."""
    num = K_sat * (phi * K0 / K_fl + 1.0 - phi) - K0
    den = phi * K0 / K_fl + K_sat / K0 - 1.0 - phi
    return num / den


def _fluid_moduli(name, k_override=None, rho_override=None):
    """Resolve a fluid's (K_fl in Pa, rho_fl in g/cc) from a preset name and/or
    explicit overrides. Overrides (K in GPa, rho in g/cc) take precedence over the
    preset. Raises ValueError if the fluid is neither a known preset nor fully
    specified by overrides, or if a resolved modulus/density is non-positive.
    """
    K = rho = None
    if name is not None:
        preset = _FLUIDS.get(str(name).lower())
        if preset is not None:
            K, rho = preset
    if k_override is not None:
        K = float(k_override) * 1e9      # GPa -> Pa
    if rho_override is not None:
        rho = float(rho_override)        # g/cc
    if K is None or rho is None:
        raise ValueError(
            f"Unknown fluid '{name}'; use one of {sorted(_FLUIDS)} "
            f"or supply k_fl/rho_fl overrides."
        )
    if K <= 0 or rho <= 0:
        raise ValueError(
            f"fluid modulus and density must be positive (got K={K} Pa, rho={rho} g/cc)"
        )
    return K, rho


def gassmann_substitution(vp, vs, rho, phi, fluid_in, fluid_out,
                          k_mineral=37.0,
                          k_fl_in=None, rho_fl_in=None,
                          k_fl_out=None, rho_fl_out=None,
                          print_results=True):
    """Gassmann fluid substitution from in-situ elastic properties.

    Args:
        vp, vs: in-situ P/S velocities (m/s), scalar or array-like.
        rho: in-situ bulk density (g/cc), scalar or array-like.
        phi: porosity (fraction, 0-1), scalar or array-like.
        fluid_in, fluid_out: 'water'/'brine'/'oil'/'gas' (case-insensitive).
        k_mineral: mineral (grain) bulk modulus in GPa (default 37, quartz).
        k_fl_in/out: optional fluid bulk-modulus override in GPa.
        rho_fl_in/out: optional fluid density override in g/cc.

    Returns:
        dict with substituted 'vp' (m/s), 'vs' (m/s), 'rho' (g/cc), 'vp_vs',
        plus 'k_dry', 'k_sat' (GPa) and 'mu' (GPa, unchanged by substitution).
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # REJECT non-physical inputs.
    if np.any(phi <= 0) or np.any(phi > 1):
        raise ValueError(
            "phi (porosity) must be within (0, 1] "
            "(zero porosity has no pore fluid to substitute)"
        )
    if np.any(vp <= 0):
        raise ValueError("vp must be positive")
    if np.any(vs <= 0):
        raise ValueError("vs must be positive")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive")
    if k_mineral is None or k_mineral <= 0:
        raise ValueError(f"k_mineral must be positive (got {k_mineral})")

    K0 = float(k_mineral) * 1e9  # GPa -> Pa
    K_fl_in, rho_fl_in_val = _fluid_moduli(fluid_in, k_fl_in, rho_fl_in)
    K_fl_out, rho_fl_out_val = _fluid_moduli(fluid_out, k_fl_out, rho_fl_out)

    # In-situ saturated moduli (SI); density g/cc -> kg/m^3.
    rho_si = rho * 1000.0
    mu = rho_si * vs ** 2                              # Pa, fluid-independent
    K_sat_in = rho_si * vp ** 2 - (4.0 / 3.0) * mu    # Pa

    # Invert to the dry frame, then forward-substitute the new fluid.
    # np.errstate suppresses numpy's bare RuntimeWarnings (e.g. sqrt of a negative
    # when inputs are inconsistent); our explicit non-physical warning below still fires.
    with np.errstate(invalid="ignore", divide="ignore"):
        K_dry = gassmann_dry(K_sat_in, K0, K_fl_in, phi)
        K_sat_out = gassmann_sat(K_dry, K0, K_fl_out, phi)

        # Density swap (only the pore fluid changes), g/cc.
        rho_out = rho + phi * (rho_fl_out_val - rho_fl_in_val)
        rho_out_si = rho_out * 1000.0

        vp_out = np.sqrt((K_sat_out + (4.0 / 3.0) * mu) / rho_out_si)
        vs_out = np.sqrt(mu / rho_out_si)
        vp_vs = vp_out / vs_out

    if np.any(K_dry < 0) or np.any(K_dry > K0):
        warnings.warn(
            "Inverted dry-frame modulus is non-physical (K_dry < 0 or > K_mineral); "
            "check vp/vs/rho/phi/k_mineral consistency.",
            stacklevel=2,
        )

    result = {
        "vp": vp_out, "vs": vs_out, "rho": rho_out, "vp_vs": vp_vs,
        "k_dry": K_dry / 1e9, "k_sat": K_sat_out / 1e9, "mu": mu / 1e9,
    }

    if print_results:
        print("\n=== Gassmann Fluid Substitution ===")
        print(f"  {fluid_in} -> {fluid_out}")
        print(f"  Vp:  {_format_value(vp_out, 0)} m/s")
        print(f"  Vs:  {_format_value(vs_out, 0)} m/s")
        print(f"  Rho: {_format_value(rho_out, 3)} g/cc")
        print(f"  Vp/Vs: {_format_value(vp_vs, 2)}")
        print("=" * 35)

    return result


def calculate_rock_properties(phit, vclay, fluid_type='water', print_results=True):
    """
    Estimate Vp, Vs, density, Vp/Vs and impedances from porosity and clay volume.

    Model: water-saturated Vp/Vs from the Han, Nur & Morgan (1986) regressions
    (clay-bearing sandstones, ~40 MPa); bulk density from mass balance; and, for
    oil/gas, proper Gassmann fluid substitution (shear modulus held
    fluid-independent, so gas LOWERS Vp but slightly RAISES Vs via lower density).

    Args:
        phit: float or array-like, porosity (fraction). Clipped to the Han
            validity range [0, 0.35].
        vclay: float or array-like, clay volume (fraction). Clipped to [0, 0.5].
        fluid_type: 'water'/'brine', 'oil', or 'gas'.
        print_results: whether to print the calculated values.

    Returns:
        tuple: (vp, vs, rhob, vp_vs_ratio, ai, si) - Vp (m/s), Vs (m/s),
               bulk density (g/cc), Vp/Vs, acoustic impedance and shear impedance
               (each in ×10⁶ kg/m²·s, i.e. MRayl).
    """
    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)

    # REJECT physically impossible fractions.
    if np.any(phit < 0) or np.any(phit > 1):
        raise ValueError("phit (porosity) must be within [0, 1]")
    if np.any(vclay < 0) or np.any(vclay > 1):
        raise ValueError("vclay (clay volume) must be within [0, 1]")

    # WARN (not silent) when outside the Han (1986) validity range, then clip.
    if np.any(phit > 0.35):
        warnings.warn("phit beyond the Han (1986) validity range (>0.35); clipping to 0.35.", stacklevel=2)
    if np.any(vclay > 0.5):
        warnings.warn("vclay beyond the Han (1986) validity range (>0.5); clipping to 0.5.", stacklevel=2)
    phit = np.clip(phit, 0.0, 0.35)
    vclay = np.clip(vclay, 0.0, 0.5)

    fluid = fluid_type.lower()
    if fluid not in _FLUIDS:
        raise ValueError(f"Unknown fluid type: {fluid_type}. Use 'water', 'oil', or 'gas'.")

    # --- Water-saturated velocities: Han, Nur & Morgan (1986), 40 MPa (km/s -> m/s) ---
    vp_w = (5.59 - 6.93 * phit - 2.18 * vclay) * 1000.0
    vs_w = (3.52 - 4.91 * phit - 1.89 * vclay) * 1000.0

    # --- Grain density and mineral bulk modulus (quartz/clay mix) ---
    rho_matrix = _RHO_QUARTZ * (1 - vclay) + _RHO_CLAY * vclay  # g/cc
    k0_voigt = (1 - vclay) * _K_QUARTZ + vclay * _K_CLAY
    k0_reuss = 1.0 / ((1 - vclay) / _K_QUARTZ + vclay / _K_CLAY)
    K0 = 0.5 * (k0_voigt + k0_reuss)  # Voigt-Reuss-Hill mineral modulus

    # --- In-situ (water-saturated) moduli ---
    K_fl_w, rho_fl_w = _FLUIDS['water']
    rho_w_sat = (rho_matrix * (1 - phit) + rho_fl_w * phit) * 1000.0  # kg/m^3
    mu = rho_w_sat * vs_w ** 2                       # shear modulus, fluid-independent
    K_sat_w = rho_w_sat * vp_w ** 2 - (4.0 / 3.0) * mu

    K_fl_t, rho_fl_t = _FLUIDS[fluid]
    if fluid in ('water', 'brine'):
        vp, vs = vp_w, vs_w
        rhob = rho_matrix * (1 - phit) + rho_fl_w * phit
    else:
        # Invert Gassmann to the dry frame, then forward-substitute the target fluid.
        K_dry = gassmann_dry(K_sat_w, K0, K_fl_w, phit)
        K_sat_t = gassmann_sat(K_dry, K0, K_fl_t, phit)
        rhob = rho_matrix * (1 - phit) + rho_fl_t * phit       # g/cc
        rho_t_sat = rhob * 1000.0
        vp = np.sqrt((K_sat_t + (4.0 / 3.0) * mu) / rho_t_sat)
        vs = np.sqrt(mu / rho_t_sat)                            # mu unchanged by fluid

    vp_vs_ratio = vp / vs

    # Calculate impedance values (density g/cc -> kg/m³)
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
        print(f"  Fluid Density: {_format_value(np.asarray(rho_fl_t, dtype=float), 3)} g/cc")
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
        
        # Add rock physics knowledge to the database. Tag with `domain` for
        # consistency with the other populate paths so domain-filtered search
        # can match these documents.
        for topic, content in ROCK_PHYSICS_KNOWLEDGE.items():
            _rock_physics_db.add_document(
                text=content,
                metadata={'domain': 'rock_physics', 'topic': topic}
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