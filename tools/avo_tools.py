# Tools for AVO analysis
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tools.physics_guards import require_elastic_medium, angles_error

def zoeppritz_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    Exact Zoeppritz PP reflection coefficient (Aki & Richards form).
    Returns a numpy array of reflection coefficients, one per incidence angle (deg).
    Post-critical angles return np.nan.
    """
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    _ang_err = angles_error(angles)
    if _ang_err:
        raise ValueError(_ang_err)
    if np.any(angles > 45):
        warnings.warn("AVO: incidence angles > 45 deg; results may be less reliable.", stacklevel=2)
    angles = np.radians(angles)
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

def _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2):
    """Shuey three-term coefficients: intercept R0, gradient G, curvature F.

    Pure (no guards, no angle). R0 is the normal-incidence reflectivity (AVO
    intercept) and G is the AVO gradient. Shared by shuey_reflectivity and
    avo_attributes so the two never diverge.
    """
    d_vp = vp2 - vp1
    d_vs = vs2 - vs1
    d_rho = rho2 - rho1
    avg_vp = 0.5 * (vp1 + vp2)
    avg_vs = 0.5 * (vs1 + vs2)
    avg_rho = 0.5 * (rho1 + rho2)
    R0 = 0.5 * (d_vp / avg_vp + d_rho / avg_rho)
    G = 0.5 * d_vp / avg_vp - 2 * (avg_vs ** 2 / avg_vp ** 2) * (d_rho / avg_rho + 2 * d_vs / avg_vs)
    F = 0.5 * d_vp / avg_vp
    return R0, G, F

_CLASS_II_INTERCEPT = 0.02  # |intercept| band that defines a Class-II response

_AVO_CLASS_DESCRIPTIONS = {
    "I": "High-impedance contrast; amplitude dims (and may reverse polarity) with offset.",
    "I*": "Atypical: positive intercept with non-negative gradient (amplitude rises with offset).",
    "II": "Near-zero intercept; weak amplitude with a phase/polarity reversal with offset.",
    "IIp": "Near-zero negative intercept; polarity reversal with offset.",
    "III": "Classic bright spot (e.g. gas sand); amplitude brightens with offset.",
    "IV": "Bright spot whose amplitude magnitude decreases with offset.",
}


def _classify_avo(intercept, gradient):
    """Return (class_label, description) from the AVO intercept and gradient.

    Rutherford & Williams (1989) / Castagna & Swan (1997), on the signs of
    A (intercept) and B (gradient) with a near-zero-intercept band for Class II.
    """
    A, B = intercept, gradient
    if abs(A) <= _CLASS_II_INTERCEPT:
        cls = "IIp" if A < 0 else "II"
    elif A > 0 and B < 0:
        cls = "I"
    elif A < 0 and B <= 0:
        # Negative intercept with a non-positive gradient (incl. flat B==0):
        # a classic bright spot that brightens (or holds) with offset.
        cls = "III"
    elif A < 0 and B > 0:
        cls = "IV"
    else:  # A > 0 and B >= 0
        cls = "I*"
    return cls, _AVO_CLASS_DESCRIPTIONS[cls]


def avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2):
    """AVO intercept, gradient, and class for a single interface.

    Args:
        vp1, vs1, rho1: P/S velocity (m/s) and density (g/cc) of the upper medium.
        vp2, vs2, rho2: P/S velocity (m/s) and density (g/cc) of the lower medium.

    Returns:
        dict: intercept (A), gradient (B), avo_class (I/I*/II/IIp/III/IV), and
        avo_class_description.
    """
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    R0, G, _ = _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2)
    cls, desc = _classify_avo(R0, G)
    return {
        "intercept": float(R0),
        "gradient": float(G),
        "avo_class": cls,
        "avo_class_description": desc,
    }


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
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    _ang_err = angles_error(angles)
    if _ang_err:
        raise ValueError(_ang_err)
    if np.any(angles > 45):
        warnings.warn("AVO: incidence angles > 45 deg; results may be less reliable.", stacklevel=2)
    angles = np.radians(angles)
    # Shuey coefficients (intercept R0, gradient G, curvature F).
    R0, G, F = _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2)
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
    # Autoscale the y-axis: a fixed +/-0.3 clip would crop bright spots / class-IV
    # anomalies. Keep RC=0 visible for reference.
    plt.axhline(0.0, color='k', linewidth=0.5, alpha=0.4)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_avo_crossplot(intercept, gradient, avo_class=None, output_path=None):
    """Plot the AVO intercept-gradient (A-B) crossplot and return the PNG path.

    Quadrants are lightly shaded and labeled by AVO class (I: A>0,B<0; III: A<0,B<0;
    IV: A<0,B>0) with a Class-II band around A=0. The (intercept, gradient) point is
    marked (annotated with avo_class when supplied).
    """
    import tempfile
    import os
    from matplotlib.patches import Rectangle

    A, B = float(intercept), float(gradient)
    # Symmetric extent that always includes the point and the origin, with a floor so
    # a near-origin point doesn't collapse the axes.
    e = max(abs(A), abs(B), 0.05) * 1.3

    fig, ax = plt.subplots(figsize=(7, 7))
    # Quadrant shading (class regions).
    ax.add_patch(Rectangle((0, -e), e, e, color="tab:blue", alpha=0.08))   # I:   A>0, B<0
    ax.add_patch(Rectangle((-e, -e), e, e, color="tab:red", alpha=0.08))   # III: A<0, B<0
    ax.add_patch(Rectangle((-e, 0), e, e, color="tab:green", alpha=0.08))  # IV:  A<0, B>0
    ax.axvspan(-_CLASS_II_INTERCEPT, _CLASS_II_INTERCEPT, color="gray", alpha=0.12)  # II band

    # Class labels at quadrant centers.
    ax.text(0.5 * e, -0.5 * e, "I", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(-0.5 * e, -0.5 * e, "III", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(-0.5 * e, 0.5 * e, "IV", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(0.0, 0.85 * e, "II", ha="center", va="center", fontsize=12, alpha=0.6)

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5)

    ax.plot(A, B, "ko", markersize=9)
    if avo_class:
        ax.annotate(f"Class {avo_class}", (A, B), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")

    ax.set_xlim(-e, e)
    ax.set_ylim(-e, e)
    ax.set_xlabel("Intercept (A)")
    ax.set_ylabel("Gradient (B)")
    ax.set_title("AVO Intercept-Gradient Crossplot")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
