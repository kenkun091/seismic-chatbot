"""OutcropInterpretation validation/normalization and lithology resolution."""
import pytest

from tools.outcrop_tools import (validate_interpretation, resolve_lithology,
                                 LITHOLOGY_TABLE, CONFIDENCE_LEVELS)


def _region(**kw):
    base = {"id": 1, "label": "sand", "lithology": "sandstone",
            "geometry": {"type": "polygon",
                         "points": [[0.1, 0.2], [0.6, 0.2], [0.6, 0.5], [0.1, 0.5]]}}
    base.update(kw)
    return base


def _interp(**kw):
    base = {"regions": [_region()],
            "scale": {"estimated_height_m": 30, "reference": "hammer", "confidence": "low"},
            "background_lithology": "shale", "mode": "polygons"}
    base.update(kw)
    return base


def test_valid_polygon_normalizes():
    out = validate_interpretation(_interp())
    r = out["regions"][0]
    assert r["id"] == 1 and r["lithology"] == "sandstone"
    assert r["geometry_type"] == "polygon" and len(r["points"]) == 4
    assert r["porosity"] is None and r["vclay"] is None
    assert r["confidence"] == "medium" and r["notes"] == ""
    assert out["scale"]["estimated_height_m"] == 30.0
    assert out["background_lithology"] == "shale"


def test_band_becomes_full_width_rectangle():
    out = validate_interpretation(_interp(
        regions=[_region(geometry={"type": "band", "y_top": 0.2, "y_bottom": 0.35})],
        mode="bands"))
    r = out["regions"][0]
    assert r["geometry_type"] == "band"
    assert r["points"] == [[0.0, 0.2], [1.0, 0.2], [1.0, 0.35], [0.0, 0.35]]


def test_missing_ids_are_assigned_sequentially():
    a = _region(); del a["id"]
    b = _region(label="b"); del b["id"]
    out = validate_interpretation(_interp(regions=[a, b]))
    assert [r["id"] for r in out["regions"]] == [1, 2]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_interpretation(_interp(regions=[_region(), _region()]))


def test_unknown_lithology_rejected():
    with pytest.raises(ValueError, match="lithology"):
        validate_interpretation(_interp(regions=[_region(lithology="kryptonite")]))


def test_lithology_is_case_and_space_insensitive():
    out = validate_interpretation(_interp(regions=[_region(lithology="Clean Sandstone")]))
    assert out["regions"][0]["lithology"] == "clean_sandstone"


def test_polygon_needs_three_points_in_unit_square():
    with pytest.raises(ValueError, match="at least 3"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "polygon", "points": [[0, 0], [1, 1]]})]))
    with pytest.raises(ValueError, match="0, 1"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "polygon", "points": [[0, 0], [1.2, 0], [1, 1]]})]))


def test_band_needs_top_above_bottom():
    with pytest.raises(ValueError, match="y_top"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "band", "y_top": 0.5, "y_bottom": 0.4})]))


def test_porosity_and_vclay_hints_validated():
    out = validate_interpretation(_interp(regions=[_region(porosity=0.22, vclay=0.05)]))
    assert out["regions"][0]["porosity"] == 0.22
    with pytest.raises(ValueError, match="porosity"):
        validate_interpretation(_interp(regions=[_region(porosity=1.5)]))


def test_scale_null_height_allowed_and_confidence_defaulted():
    out = validate_interpretation(_interp(scale={"estimated_height_m": None}))
    assert out["scale"] == {"estimated_height_m": None, "reference": None,
                            "confidence": "low"}


def test_scale_bad_confidence_rejected():
    with pytest.raises(ValueError, match="confidence"):
        validate_interpretation(_interp(scale={"estimated_height_m": 10,
                                               "confidence": "certain"}))


def test_background_cannot_be_cover():
    with pytest.raises(ValueError, match="background"):
        validate_interpretation(_interp(background_lithology="cover"))


def test_regions_must_be_list():
    with pytest.raises(ValueError, match="regions"):
        validate_interpretation({"regions": "none"})


def test_table_has_both_routes_and_cover():
    routes = {v["route"] for v in LITHOLOGY_TABLE.values()}
    assert routes == {"han", "direct", "background"}
    assert LITHOLOGY_TABLE["cover"]["route"] == "background"
    assert "low" in CONFIDENCE_LEVELS


def test_resolve_han_route_matches_predict_layer():
    from workflows.adapters import predict_layer
    got = resolve_lithology("sandstone")
    exp = predict_layer(0.20, 0.10, fluid="brine", label="sandstone")
    assert got["route"] == "han"
    assert got["vp"] == pytest.approx(exp.vp) and got["vs"] == pytest.approx(exp.vs)
    assert got["fluid"] == "brine" and got["phit"] == 0.20


def test_resolve_han_route_with_gas_lowers_vp():
    brine = resolve_lithology("sandstone", fluid="brine")
    gas = resolve_lithology("sandstone", fluid="gas")
    assert gas["vp"] < brine["vp"] and gas["vs"] > brine["vs"]


def test_resolve_direct_route_returns_table_values():
    got = resolve_lithology("limestone")
    assert got == {"vp": 5000.0, "vs": 2700.0, "rho": 2.55, "route": "direct",
                   "phit": None, "vclay": None, "fluid": None}


def test_resolve_direct_route_rejects_petro_overrides():
    with pytest.raises(ValueError, match="limestone"):
        resolve_lithology("limestone", fluid="gas")
    with pytest.raises(ValueError, match="limestone"):
        resolve_lithology("limestone", porosity=0.1)


def test_resolve_cover_rejected():
    with pytest.raises(ValueError, match="cover"):
        resolve_lithology("cover")
