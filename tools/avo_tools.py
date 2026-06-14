# Tools for AVO analysis
import numpy as np
import matplotlib.pyplot as plt

def zoeppritz_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    Exact Zoeppritz PP reflection coefficient (Aki & Richards form).
    Returns a numpy array of reflection coefficients, one per incidence angle (deg).
    Post-critical angles return np.nan.
    """
    angles = np.radians(np.asarray(angles, dtype=float))
    rc = []
    for theta1 in np.atleast_1d(angles):
        p = np.sin(theta1) / vp1  # horizontal slowness (ray parameter)
        # Snell's law; bail out post-critical for the transmitted P-wave
        if abs(p * vp2) > 1.0:
            rc.append(np.nan)
            continue
        theta2 = np.arcsin(p * vp2)
        phi1 = np.arcsin(np.clip(p * vs1, -1.0, 1.0))
        phi2 = np.arcsin(np.clip(p * vs2, -1.0, 1.0))

        a = rho2 * (1 - 2 * np.sin(phi2) ** 2) - rho1 * (1 - 2 * np.sin(phi1) ** 2)
        b = rho2 * (1 - 2 * np.sin(phi2) ** 2) + 2 * rho1 * np.sin(phi1) ** 2
        c = rho1 * (1 - 2 * np.sin(phi1) ** 2) + 2 * rho2 * np.sin(phi2) ** 2
        d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

        E = b * np.cos(theta1) / vp1 + c * np.cos(theta2) / vp2
        F = b * np.cos(phi1) / vs1 + c * np.cos(phi2) / vs2
        G = a - d * (np.cos(theta1) / vp1) * (np.cos(phi2) / vs2)
        H = a - d * (np.cos(theta2) / vp2) * (np.cos(phi1) / vs1)

        D = E * F + G * H * p ** 2

        rpp = (
            ((b * np.cos(theta1) / vp1 - c * np.cos(theta2) / vp2) * F
             - (a + d * (np.cos(theta1) / vp1) * (np.cos(phi2) / vs2)) * H * p ** 2)
            / D
        )
        rc.append(rpp)
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

def plot_avo_reflectivity(angles, rc, output_path=None):
    """
    Plot AVO reflectivity curve and return the path to the plot.
    
    Args:
        angles: array-like, incident angles in degrees
        rc: array-like, reflection coefficients
        output_path: Optional path to save the plot. If None, creates a temporary file.
        
    Returns:
        str: Path to the saved plot file
    """
    import tempfile
    import os
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, rc, 'b-', linewidth=2, label='Reflection Coefficient')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Reflection Coefficient')
    plt.title('AVO Reflectivity Curve')
    plt.ylim(-0.3,0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
