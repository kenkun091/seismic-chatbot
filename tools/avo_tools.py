# Tools for AVO analysis
import os
import tempfile
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

def extended_elastic_impedance(vp, vs, rho, chi,
                               vp0=None, vs0=None, rho0=None, k=None):
    """Extended Elastic Impedance, EEI(chi) (Whitcombe, 2002).

    EEI(chi) = Vp0*rho0 * (Vp/Vp0)^p * (Vs/Vs0)^q * (rho/rho0)^r
        p = cos(chi) + sin(chi);  q = -8K*sin(chi);  r = cos(chi) - 4K*sin(chi)
        K = (Vs/Vp)^2  (background; override with `k`)
    chi is the rotation angle in degrees. At chi=0, EEI = Vp*rho (acoustic impedance).
    Raw EEI (reference Vp0=Vs0=rho0=1) is returned unless ALL of vp0/vs0/rho0 are
    supplied, in which case Whitcombe normalization is applied.

    Args:
        vp, vs, rho: scalar layer P/S velocity (m/s) and density (g/cc).
        chi: array-like rotation angles in degrees (|chi| <= 90).
        vp0, vs0, rho0: optional reference constants (all-or-nothing).
        k: optional background (Vs/Vp)^2; default computed from vs/vp.

    Returns:
        np.ndarray of EEI values, one per chi.
    """
    require_elastic_medium(vp, vs, rho)

    refs = (vp0, vs0, rho0)
    n_set = sum(ref is not None for ref in refs)
    if n_set not in (0, 3):
        raise ValueError(
            "reference constants vp0/vs0/rho0 are all-or-nothing: supply all three "
            "(Whitcombe normalization) or none (raw EEI)."
        )
    if n_set == 3:
        for name, ref in (("vp0", vp0), ("vs0", vs0), ("rho0", rho0)):
            if ref <= 0:
                raise ValueError(f"{name} must be positive (got {ref})")
        rvp, rvs, rrho = float(vp0), float(vs0), float(rho0)
    else:
        rvp = rvs = rrho = 1.0

    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    if np.any(np.abs(chi) > 90):
        raise ValueError("chi (rotation angle) must be within [-90, 90] degrees")

    K = (vs / vp) ** 2 if k is None else float(k)
    x = np.radians(chi)
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)

    return rvp * rrho * (vp / rvp) ** p * (vs / rvs) ** q * (rho / rrho) ** r


def plot_extended_elastic_impedance(chi, eei, output_path=None):
    """Plot EEI vs rotation angle chi and return the PNG path.

    Marks chi=0 (where EEI equals the acoustic impedance) for reference.
    """
    chi = np.asarray(chi, dtype=float)
    eei = np.asarray(eei, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, eei, "b-", linewidth=2, label="EEI(χ)")
    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5, label="χ=0 (AI)")
    ax.set_xlabel("Rotation angle χ (degrees)")
    ax.set_ylabel("Extended Elastic Impedance")
    ax.set_title("Extended Elastic Impedance vs χ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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


def _eei_chi_scan(vp, vs, rho, target, chi, k=None):
    """Scan rotation angle chi for the EEI projection best correlated with a target log.

    EEI(chi) over a log (Whitcombe 2002), with a single SCALAR background K so chi has
    a consistent meaning across the interval. Returns chi*, the Pearson r vs chi curve,
    the signed correlation at chi*, and the EEI log at chi*. Raw (un-normalized) EEI is
    used: Pearson r is scale-invariant, so normalization is unnecessary.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    target = np.asarray(target, dtype=float)
    chi = np.atleast_1d(np.asarray(chi, dtype=float))

    if not (vp.shape == vs.shape == rho.shape == target.shape) or vp.ndim != 1:
        raise ValueError("vp, vs, rho, target must be 1-D logs of equal length")
    if vp.size < 2:
        raise ValueError("logs must have at least 2 samples to correlate")
    if chi.size == 0:
        raise ValueError("chi sweep is empty")
    if np.any(np.abs(chi) > 90):
        raise ValueError("chi (rotation angle) must be within [-90, 90] degrees")
    # Per-sample physical validity (vp>0, rho>0, 0<vs<vp).
    if np.any(vp <= 0) or np.any(rho <= 0) or np.any(vs <= 0) or np.any(vs >= vp):
        raise ValueError("non-physical elastic sample: require vp>0, rho>0, 0<vs<vp")
    if np.std(target) == 0:
        raise ValueError("target log has zero variance; cannot correlate")

    K = float(np.mean((vs / vp) ** 2)) if k is None else float(k)
    x = np.radians(chi)
    # exponents per chi (1-D, length n_chi)
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)

    # EEI log per chi: shape (n_samples, n_chi) via outer broadcasting.
    log_eei = (np.log(vp)[:, None] * p[None, :]
               + np.log(vs)[:, None] * q[None, :]
               + np.log(rho)[:, None] * r[None, :])
    eei = np.exp(log_eei)  # (n_samples, n_chi)

    t = target - target.mean()
    t_norm = np.sqrt(np.sum(t ** 2))
    e = eei - eei.mean(axis=0, keepdims=True)
    e_norm = np.sqrt(np.sum(e ** 2, axis=0))
    e_norm = np.where(e_norm == 0, np.nan, e_norm)  # guard flat EEI columns
    corr = (t @ e) / (t_norm * e_norm)  # Pearson r per chi, shape (n_chi,)

    best = int(np.nanargmax(np.abs(corr)))
    return {
        "chi": [float(c) for c in chi],
        "correlation": [float(c) for c in corr],
        "optimal_chi": float(chi[best]),
        "max_correlation": float(corr[best]),
        "eei_optimal": [float(v) for v in eei[:, best]],
    }


def plot_eei_chi_scan(chi, correlation, optimal_chi, output_path=None):
    """Plot Pearson correlation vs rotation angle chi, marking the optimal chi."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    chi = np.asarray(chi, dtype=float)
    correlation = np.asarray(correlation, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, correlation, "b-", linewidth=2, label="Pearson r(χ)")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.axvline(optimal_chi, color="r", lw=1.2, ls="--",
               label=f"χ* = {optimal_chi:.1f}°")
    ax.set_xlabel("Rotation angle χ (degrees)")
    ax.set_ylabel("Correlation with target")
    ax.set_title("EEI–target correlation vs χ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def eei_optimal_chi(vp, vs, rho, target, chi_min=-90.0, chi_max=90.0,
                    chi_step=1.0, k=None):
    """Find the EEI rotation angle chi best correlated with a target log (raw-logs mode).

    Sweeps chi in [chi_min, chi_max] (step chi_step), correlates EEI(chi) against the
    target log, and returns chi*, the correlation curve, the EEI log at chi*, and a
    correlation-vs-chi plot path.
    """
    chi = np.arange(chi_min, chi_max + chi_step, chi_step)
    result = _eei_chi_scan(vp, vs, rho, target, chi, k=k)
    result["image_path"] = plot_eei_chi_scan(
        result["chi"], result["correlation"], result["optimal_chi"]
    )
    return result
