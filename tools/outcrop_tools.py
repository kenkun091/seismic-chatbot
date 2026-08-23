"""Outcrop photo -> facies interpretation -> 2-D elastic earth model.

Pipeline (each step is a registry tool; results hand off via ContextManager):
  interpret_outcrop     photo -> OutcropInterpretation (vision LLM; the ONLY
                        function here that touches a network)
  outcrop_to_model      interpretation + scale + lithology table -> EarthModel2D
The generic 2-D convolution lives in tools/section_tools.py and knows nothing
about outcrops.
"""
import json
import os
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from tools.image_safety import safe_image_path, downscale_for_vision, image_size
from workflows.adapters import predict_layer

CONFIDENCE_LEVELS = ("low", "medium", "high")

# Han (1986)/Gassmann route for clastics; literature "typical" values (Mavko et
# al., Rock Physics Handbook; Bourbie et al.) for rocks Han does not model.
# Shale vclay is 0.50 — the ceiling of Han's calibrated clay range — so the
# default background never triggers warn-and-clip.
LITHOLOGY_TABLE: Dict[str, Dict[str, Any]] = {
    "shale":           {"route": "han", "phit": 0.10, "vclay": 0.50, "fluid": "brine"},
    "mudstone":        {"route": "han", "phit": 0.10, "vclay": 0.50, "fluid": "brine"},
    "siltstone":       {"route": "han", "phit": 0.15, "vclay": 0.40, "fluid": "brine"},
    "sandstone":       {"route": "han", "phit": 0.20, "vclay": 0.10, "fluid": "brine"},
    "clean_sandstone": {"route": "han", "phit": 0.25, "vclay": 0.02, "fluid": "brine"},
    "conglomerate":    {"route": "han", "phit": 0.15, "vclay": 0.05, "fluid": "brine"},
    "limestone":       {"route": "direct", "vp": 5000.0, "vs": 2700.0, "rho": 2.55},
    "dolomite":        {"route": "direct", "vp": 5800.0, "vs": 3200.0, "rho": 2.75},
    "chalk":           {"route": "direct", "vp": 3500.0, "vs": 1900.0, "rho": 2.20},
    "salt":            {"route": "direct", "vp": 4500.0, "vs": 2600.0, "rho": 2.10},
    "coal":            {"route": "direct", "vp": 2400.0, "vs": 1200.0, "rho": 1.40},
    "basalt":          {"route": "direct", "vp": 5500.0, "vs": 3100.0, "rho": 2.80},
    "cover":           {"route": "background"},
}

LITHOLOGY_COLORS = {
    "shale": "#6b705c", "mudstone": "#7a7d6e", "siltstone": "#b5a37a",
    "sandstone": "#e9c46a", "clean_sandstone": "#f4e285", "conglomerate": "#d4a373",
    "limestone": "#8ecae6", "dolomite": "#6a9fb5", "chalk": "#dbe9ee",
    "salt": "#f2b5d4", "coal": "#222222", "basalt": "#5c4b51", "cover": "#9ccc65",
}


def _norm_lithology(name: Any) -> str:
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    if key not in LITHOLOGY_TABLE:
        raise ValueError(
            f"unknown lithology {name!r}; use one of {sorted(LITHOLOGY_TABLE)}"
        )
    return key


def _opt_fraction(value: Any, name: str, rid: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"region {rid}: {name} must be a number in [0, 1] (got {value!r})")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"region {rid}: {name} must be in [0, 1] (got {v})")
    return v


def _confidence(value: Any, default: str, where: str) -> str:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v not in CONFIDENCE_LEVELS:
        raise ValueError(f"{where}: confidence must be one of {CONFIDENCE_LEVELS} (got {value!r})")
    return v


def _points_from_geometry(geom: Any, rid: Any) -> Tuple[str, List[List[float]]]:
    if not isinstance(geom, dict):
        raise ValueError(f"region {rid}: geometry must be an object")
    gtype = str(geom.get("type", "polygon")).lower()
    if gtype == "band":
        try:
            y_top = float(geom["y_top"]); y_bot = float(geom["y_bottom"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"region {rid}: band geometry needs numeric y_top and y_bottom")
        if not (0.0 <= y_top < y_bot <= 1.0):
            raise ValueError(f"region {rid}: band needs 0 <= y_top < y_bottom <= 1 "
                             f"(got y_top={y_top}, y_bottom={y_bot})")
        return "band", [[0.0, y_top], [1.0, y_top], [1.0, y_bot], [0.0, y_bot]]
    if gtype != "polygon":
        raise ValueError(f"region {rid}: geometry type must be 'polygon' or 'band' (got {gtype!r})")
    pts = geom.get("points")
    if not isinstance(pts, list) or len(pts) < 3:
        raise ValueError(f"region {rid}: polygon needs at least 3 points")
    out = []
    for p in pts:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"region {rid}: each point must be [x, y]")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"region {rid}: point coordinates must be in [0, 1] (got {p})")
        out.append([x, y])
    return "polygon", out


def validate_interpretation(data: Any) -> Dict[str, Any]:
    """Validate + normalize a raw OutcropInterpretation dict (see spec).

    Returns a new dict; raises ValueError naming the offending field. Bands are
    converted to full-width rectangles so rasterization has one geometry path.
    """
    if not isinstance(data, dict):
        raise ValueError("interpretation must be a JSON object")
    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("interpretation.regions must be a list")

    regions = []
    seen_ids = set()
    next_id = 1
    for i, r in enumerate(raw_regions):
        if not isinstance(r, dict):
            raise ValueError(f"regions[{i}] must be an object")
        rid = r.get("id")
        if rid is None:
            rid = next_id
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            raise ValueError(f"regions[{i}]: id must be an integer (got {r.get('id')!r})")
        if rid in seen_ids:
            raise ValueError(f"duplicate region id {rid}")
        seen_ids.add(rid)
        next_id = max(next_id, rid + 1)
        lith = _norm_lithology(r.get("lithology"))
        gtype, pts = _points_from_geometry(r.get("geometry"), rid)
        regions.append({
            "id": rid,
            "label": str(r.get("label") or lith),
            "lithology": lith,
            "geometry_type": gtype,
            "points": pts,
            "porosity": _opt_fraction(r.get("porosity"), "porosity", rid),
            "vclay": _opt_fraction(r.get("vclay"), "vclay", rid),
            "confidence": _confidence(r.get("confidence"), "medium", f"region {rid}"),
            "notes": str(r.get("notes") or ""),
        })

    raw_scale = data.get("scale") or {}
    if not isinstance(raw_scale, dict):
        raise ValueError("interpretation.scale must be an object")
    height = raw_scale.get("estimated_height_m")
    if height is not None:
        try:
            height = float(height)
        except (TypeError, ValueError):
            raise ValueError("scale.estimated_height_m must be a number or null")
        if height <= 0:
            raise ValueError("scale.estimated_height_m must be positive")
    ref = raw_scale.get("reference")
    scale = {
        "estimated_height_m": height,
        "reference": (str(ref) if ref else None),
        "confidence": _confidence(raw_scale.get("confidence"), "low", "scale"),
    }

    background = _norm_lithology(data.get("background_lithology") or "shale")
    if LITHOLOGY_TABLE[background]["route"] == "background":
        raise ValueError("background_lithology cannot be 'cover'")

    mode = str(data.get("mode") or "polygons").lower()
    if mode not in ("polygons", "bands"):
        raise ValueError("mode must be 'polygons' or 'bands'")

    out = {"regions": regions, "scale": scale,
           "background_lithology": background, "mode": mode}
    for passthrough in ("image_path", "image_size", "summary"):
        if passthrough in data:
            out[passthrough] = data[passthrough]
    return out


def resolve_lithology(lithology: str, porosity: Optional[float] = None,
                      vclay: Optional[float] = None,
                      fluid: Optional[str] = None) -> Dict[str, Any]:
    """Lithology (+ optional petro overrides) -> {vp, vs, rho, route, phit, vclay, fluid}.

    Han route: predict_layer(phit, vclay, fluid). Direct route: table values;
    any of porosity/vclay/fluid raises (Han/Gassmann is not valid for them).
    """
    key = _norm_lithology(lithology)
    entry = LITHOLOGY_TABLE[key]
    route = entry["route"]
    if route == "background":
        raise ValueError("'cover' is not a rock; it is rasterized as the background lithology")
    if route == "direct":
        if porosity is not None or vclay is not None or fluid is not None:
            raise ValueError(
                f"{key} uses fixed literature Vp/Vs/density; porosity, vclay and fluid "
                f"overrides only apply to clastic (Han/Gassmann) lithologies"
            )
        return {"vp": float(entry["vp"]), "vs": float(entry["vs"]),
                "rho": float(entry["rho"]), "route": "direct",
                "phit": None, "vclay": None, "fluid": None}
    phit = float(entry["phit"] if porosity is None else porosity)
    vcl = float(entry["vclay"] if vclay is None else vclay)
    fl = str(fluid or entry["fluid"]).lower()
    layer = predict_layer(phit, vcl, fluid=fl, label=key)
    return {"vp": float(layer.vp), "vs": float(layer.vs), "rho": float(layer.rho),
            "route": "han", "phit": phit, "vclay": vcl, "fluid": fl}


# ---------------------------------------------------------------------------
# Vision interpretation
# ---------------------------------------------------------------------------

_ROCK_NAMES = sorted(k for k, v in LITHOLOGY_TABLE.items() if v["route"] != "background")

OUTCROP_PROMPT = f"""You are a field geologist interpreting an outcrop photograph for seismic forward modeling.

Return ONLY a JSON object (no prose, no markdown fences) with this exact shape:
{{
  "regions": [
    {{"id": 1, "label": "short name", "lithology": "<one of: {', '.join(_ROCK_NAMES)}, cover>",
     "geometry": {{"type": "polygon", "points": [[x, y], [x, y], [x, y]]}},
     "porosity": 0.2, "vclay": 0.1, "confidence": "low|medium|high", "notes": "texture, bedding"}}
  ],
  "scale": {{"estimated_height_m": 30, "reference": "what you measured against", "confidence": "low|medium|high"}},
  "background_lithology": "shale",
  "mode": "polygons"
}}

Rules:
- Coordinates are fractions of the image: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom).
- Outline every distinct rock body or bed as a polygon (3+ points, clockwise). If the exposure is a simple
  horizontal layer-cake, you may instead use "mode": "bands" with geometry {{"type": "band", "y_top": 0.2, "y_bottom": 0.35}}.
- Everything you do not outline is treated as the background lithology (default shale).
- Mark sky, vegetation, soil, talus, water, people and equipment as lithology "cover" so they are ignored.
- "porosity" and "vclay" are optional fractions (0-1); include them only when texture (grain size, sorting,
  cementation, mud content) justifies a value different from a typical rock of that lithology.
- Scale: look for a scale bar, hammer (~0.3 m), person (~1.7 m), lens cap, vehicle, or any labelled dimension,
  and estimate the total height of the photographed exposure in metres. If nothing gives a reference, set
  "estimated_height_m": null and "confidence": "low". Never invent a scale.
- Use integer ids starting at 1. Keep labels short.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first {...} JSON object out of model text (fences/prose tolerated)."""
    if not isinstance(text, str):
        raise ValueError("vision model returned no text")
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the vision model's answer")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"vision model answer is not valid JSON: {exc}")


def _summarize(interp: Dict[str, Any]) -> str:
    rocks = [r for r in interp["regions"] if r["lithology"] != "cover"]
    parts = [f"{len(rocks)} rock region(s) on a {interp['background_lithology']} background"]
    for r in rocks:
        parts.append(f"#{r['id']} {r['label']} ({r['lithology']}, {r['confidence']})")
    s = interp["scale"]
    if s["estimated_height_m"] is None:
        parts.append("scale: none found — please give the outcrop height in metres")
    else:
        parts.append(f"scale: ~{s['estimated_height_m']:g} m high from {s['reference'] or 'unknown reference'} "
                     f"({s['confidence']} confidence)")
    return "; ".join(parts)


def interpret_outcrop(image_path: Optional[str] = None, vision_client=None,
                      upload_dir: Optional[str] = None) -> Dict[str, Any]:
    """Photo -> validated OutcropInterpretation via the vision LLM (one retry).

    `image_path` is filled by the chatbot from the uploaded photo when omitted.
    The user's free text is never injected into the vision prompt; guidance
    goes through outcrop_to_model(overrides=...).
    """
    if not image_path:
        raise ValueError("Please upload an outcrop photo first.")
    base = upload_dir or SEISMIC_UPLOAD_DIR
    path = safe_image_path(image_path, base, MAX_IMAGE_MB)
    if vision_client is None:
        from core.vision_client import build_vision_client
        vision_client = build_vision_client()

    img_bytes, mime = downscale_for_vision(path)
    prompt = OUTCROP_PROMPT
    last_err = None
    interp = None
    for _attempt in range(2):
        text = vision_client.interpret_image(img_bytes, mime, prompt)
        try:
            interp = validate_interpretation(extract_json(text))
            break
        except ValueError as exc:
            last_err = exc
            prompt = (OUTCROP_PROMPT
                      + f"\n\nYour previous answer was invalid: {exc}\n"
                        "Return only the corrected JSON object.")
    if interp is None:
        raise ValueError(f"could not interpret image: {last_err}")

    w, h = image_size(path)
    interp["image_path"] = path
    interp["image_size"] = [int(w), int(h)]
    interp["summary"] = _summarize(interp)
    return interp


def plot_outcrop_interpretation(interpretation: Dict[str, Any],
                                output_path: Optional[str] = None) -> str:
    """Photo with semi-transparent facies polygons, ids, legend and scale note."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    from PIL import Image
    with Image.open(interpretation["image_path"]) as im:
        img = np.asarray(im.convert("RGB"))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10 * h / float(w)))
    ax.imshow(img)
    used = {}
    for r in interpretation["regions"]:
        color = LITHOLOGY_COLORS.get(r["lithology"], "#ff00ff")
        pts = np.array(r["points"]) * [w, h]
        ax.add_patch(MplPolygon(pts, closed=True, facecolor=color, edgecolor="k",
                                alpha=0.35 if r["lithology"] != "cover" else 0.15, lw=1.2))
        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, f"#{r['id']} {r['label']}", ha="center", va="center",
                fontsize=8, color="k",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        used[r["lithology"]] = color
    used.setdefault(interpretation["background_lithology"],
                    LITHOLOGY_COLORS[interpretation["background_lithology"]])
    handles = [MplPolygon([[0, 0], [1, 0], [1, 1]], facecolor=c, edgecolor="k", alpha=0.5,
                          label=n) for n, c in used.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    s = interpretation["scale"]
    scale_txt = ("scale: not found" if s["estimated_height_m"] is None
                 else f"~{s['estimated_height_m']:g} m high ({s['confidence']}, {s['reference'] or '?'})")
    ax.set_title(f"Outcrop interpretation — background {interpretation['background_lithology']}; {scale_txt}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
