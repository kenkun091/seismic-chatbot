# Tools for AVO analysis

import numpy as np

def zoeppritz_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    Compute PP reflection coefficients using the full Zoeppritz equations.
    Args:
        vp1, vs1, rho1: P-wave velocity, S-wave velocity, and density of upper layer
        vp2, vs2, rho2: P-wave velocity, S-wave velocity, and density of lower layer
        angles: array-like, incident angles in degrees
    Returns:
        numpy array of reflection coefficients for each angle
    """
    angles = np.radians(np.asarray(angles))
    rc = []
    for theta1 in angles:
        # Snell's law
        sin_theta2 = vp1 / vp2 * np.sin(theta1)
        if np.abs(sin_theta2) > 1:
            rc.append(np.nan)
            continue
        theta2 = np.arcsin(sin_theta2)
        phi1 = np.arcsin(vp1 / vs1 * np.sin(theta1)) if vs1 > 0 else 0
        phi2 = np.arcsin(vp2 / vs2 * np.sin(theta2)) if vs2 > 0 else 0
        # Zoeppritz matrix elements (simplified for PP)
        a = rho2 * (1 - 2 * (vs2 / vp2 * np.sin(theta2)) ** 2) - rho1 * (1 - 2 * (vs1 / vp1 * np.sin(theta1)) ** 2)
        b = rho2 * (1 - 2 * (vs2 / vp2 * np.sin(theta2)) ** 2) + 2 * rho1 * (vs1 / vp1 * np.sin(theta1)) ** 2
        c = rho1 * (1 - 2 * (vs1 / vp1 * np.sin(theta1)) ** 2) + 2 * rho2 * (vs2 / vp2 * np.sin(theta2)) ** 2
        d = 0.5 * (b / vp1 + c / vp2)
        r = a / d if d != 0 else np.nan
        rc.append(r)
    return np.array(rc)

def shuey_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    Compute PP reflection coefficients using the Shuey approximation.
    Args:
        vp1, vs1, rho1: P-wave velocity, S-wave velocity, and density of upper layer
        vp2, vs2, rho2: P-wave velocity, S-wave velocity, and density of lower layer
        angles: array-like, incident angles in degrees
    Returns:
        numpy array of reflection coefficients for each angle
    """
    angles = np.radians(np.asarray(angles))
    # Shuey coefficients
    d_vp = vp2 - vp1
    d_vs = vs2 - vs1
    d_rho = rho2 - rho1
    avg_vp = 0.5 * (vp1 + vp2)
    avg_vs = 0.5 * (vs1 + vs2)
    avg_rho = 0.5 * (rho1 + rho2)
    R0 = 0.5 * (d_vp / avg_vp + d_rho / avg_rho)
    G = 0.5 * d_vp / avg_vp - 2 * (avg_vs ** 2 / avg_vp ** 2) * (d_rho / avg_rho + 2 * d_vs / avg_vs)
    F = 0.5 * d_vp / avg_vp
    rc = R0 + G * np.sin(angles) ** 2 + F * (np.tan(angles) ** 2 - np.sin(angles) ** 2)
    return rc

def avo_fluid_indicator(intercept, gradient):
    """
    Simple fluid indicator based on AVO intercept and gradient.
    Args:
        intercept: AVO intercept (R0)
        gradient: AVO gradient (G)
    Returns:
        str: Fluid indicator ('gas', 'brine', 'oil', or 'uncertain')
    """
    # Example rule: negative intercept and large negative gradient suggests gas
    if intercept < 0 and gradient < -0.1:
        return 'gas'
    elif intercept > 0 and gradient > 0:
        return 'brine'
    elif intercept > 0 and gradient < 0:
        return 'oil'
    else:
        return 'uncertain'