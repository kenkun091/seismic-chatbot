"""JSON payload builders for the session API.

Tool results carry numpy arrays and server paths; the browser client needs
plain JSON with bounded precision. Values only — never paths."""
import math
from typing import Any, Dict, Optional

import numpy as np

_MODEL_SUMMARY_KEYS = ("height_m", "width_m", "image_top_m", "dz", "dx", "nz", "nx",
                       "pad_m", "scale_source", "scale_confidence",
                       "background_lithology", "legend", "regions")
_PHOTO_KEYS = ("image_top_m", "height_m", "width_m")


def _round_sig(x: float, sig: int) -> Optional[float]:
    if math.isnan(x) or math.isinf(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{sig}g}")


def to_jsonable(value: Any, sig: int = 4) -> Any:
    """Recursively convert numpy/tuples/floats into JSON-native values."""
    if isinstance(value, np.ndarray):
        return [to_jsonable(v, sig) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, sig) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, sig) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _round_sig(float(value), sig)
    return str(value)


def model_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    """Scalars + legend + region provenance of an outcrop_to_model result; no grids."""
    return {k: to_jsonable(model.get(k)) for k in _MODEL_SUMMARY_KEYS}


def section_payload(last_section: Dict[str, Any],
                    model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Depth-domain section as columns plus the photo extent for overlay registration."""
    params = last_section["parameters"]
    if params.get("domain") != "depth":
        raise ValueError("section must be computed in the depth domain for the overlay")
    section = np.asarray(last_section["section"], dtype=float)
    out: Dict[str, Any] = {
        "z": to_jsonable(np.asarray(last_section["axis"], dtype=float)),
        "traces": to_jsonable(section.T),
        "domain": "depth",
    }
    for k in ("dx", "nx", "wavelet_freq", "angle", "method", "max_abs_amplitude"):
        out[k] = to_jsonable(params.get(k))
    for k in _PHOTO_KEYS:
        out[k] = to_jsonable(model.get(k)) if isinstance(model, dict) else None
    return out


def interpretation_caps(data: Any, max_regions: int = 200, max_points: int = 2000) -> None:
    """Bound the rasterization cost of a client-supplied interpretation."""
    if not isinstance(data, dict):
        return
    regions = data.get("regions")
    if isinstance(regions, list):
        if len(regions) > max_regions:
            raise ValueError(f"too many regions ({len(regions)} > {max_regions})")
        for r in regions:
            pts = r.get("points") if isinstance(r, dict) else None
            if isinstance(pts, list) and len(pts) > max_points:
                raise ValueError(f"too many points in a region ({len(pts)} > {max_points})")
