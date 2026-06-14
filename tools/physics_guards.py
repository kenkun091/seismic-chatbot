"""Physical-validity guards for seismic compute inputs.

Two tiers:
- REJECT physically impossible inputs (raise ValueError via the require_* helpers).
- WARN on possible-but-out-of-range / numerically risky inputs (warnings.warn).

Pure and dependency-free (no numpy) so it is trivially testable and importable
from any tool. Callers pass plain floats / iterables of floats.
"""
import warnings
from typing import Optional


def elastic_medium_error(vp, vs, rho, label="medium") -> Optional[str]:
    """Return an error string if (vp, vs, rho) is not a physical elastic solid.

    Requires vp > 0, rho > 0, and 0 < vs < vp (which also keeps Poisson's ratio
    in (-1, 0.5)). Returns None when valid.
    """
    if vp is None or vp <= 0:
        return f"{label}: vp must be positive (got {vp})"
    if rho is None or rho <= 0:
        return f"{label}: density must be positive (got {rho})"
    if vs is None or vs <= 0:
        return f"{label}: vs must be positive (got {vs})"
    if vs >= vp:
        return f"{label}: vs must be less than vp (got vs={vs}, vp={vp})"
    return None


def positive_error(value, name) -> Optional[str]:
    """Return an error string if value is missing or non-positive, else None."""
    if value is None or value <= 0:
        return f"{name} must be positive (got {value})"
    return None


def angles_error(angles) -> Optional[str]:
    """Return an error string if any incidence angle is outside [0, 90) deg."""
    for a in angles:
        if a < 0 or a >= 90:
            return f"incidence angle must be in [0, 90) degrees (got {a})"
    return None


def require_elastic_medium(vp, vs, rho, label="medium") -> None:
    err = elastic_medium_error(vp, vs, rho, label)
    if err:
        raise ValueError(err)


def require_positive(value, name) -> None:
    err = positive_error(value, name)
    if err:
        raise ValueError(err)


def warn_if_aliased(max_content_hz, dt_seconds, label="wavelet") -> None:
    """Warn if frequency content reaches/exceeds Nyquist (0.5 / dt_seconds)."""
    if dt_seconds and dt_seconds > 0:
        nyquist = 0.5 / dt_seconds
        if max_content_hz >= nyquist:
            warnings.warn(
                f"{label}: frequency content (~{max_content_hz:g} Hz) reaches or exceeds "
                f"the Nyquist frequency ({nyquist:g} Hz) for dt={dt_seconds:g} s; "
                f"results may be aliased.",
                stacklevel=2,
            )


def warn_if_outside(value, lo, hi, name, unit="") -> None:
    """Warn (and proceed) if value is outside [lo, hi]."""
    if value < lo or value > hi:
        u = f" {unit}" if unit else ""
        warnings.warn(
            f"{name}={value:g}{u} is outside the expected range [{lo:g}, {hi:g}]{u}.",
            stacklevel=2,
        )
