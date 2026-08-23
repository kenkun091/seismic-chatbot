"""outcrop_to_model: scale policy, overrides, rasterization, padding, provenance."""
import numpy as np
import pytest

from tools.outcrop_tools import (outcrop_to_model, apply_overrides,
                                 validate_interpretation, resolve_lithology)


def _interp(regions, height=25.0, image_size=(400, 200)):
    d = {"regions": regions,
         "scale": {"estimated_height_m": height, "reference": "person", "confidence": "medium"},
         "background_lithology": "shale", "mode": "polygons",
         "image_size": list(image_size)}
    return validate_interpretation(d)


BAND = {"id": 1, "label": "sand bed", "lithology": "sandstone",
        "geometry": {"type": "band", "y_top": 0.2, "y_bottom": 0.4}}
LENS = {"id": 2, "label": "lime lens", "lithology": "limestone",
        "geometry": {"type": "polygon",
                     "points": [[0.25, 0.6], [0.75, 0.6], [0.75, 0.9], [0.25, 0.9]]}}
COVER = {"id": 3, "label": "bush", "lithology": "cover",
         "geometry": {"type": "polygon", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]]}}


def test_scale_from_vision_when_height_not_given():
    m = outcrop_to_model(_interp([BAND]))
    assert m["height_m"] == 25.0 and m["scale_source"] == "vision"
    assert m["scale_confidence"] == "medium"


def test_explicit_height_overrides_vision():
    m = outcrop_to_model(_interp([BAND]), height_m=40.0)
    assert m["height_m"] == 40.0 and m["scale_source"] == "user"


def test_no_scale_anywhere_asks_for_height():
    with pytest.raises(ValueError, match="height in metres"):
        outcrop_to_model(_interp([BAND], height=None))


def test_missing_interpretation_asks_to_interpret_first():
    with pytest.raises(ValueError, match="interpret_outcrop"):
        outcrop_to_model(None, height_m=10)


def test_grid_geometry_follows_aspect_and_targets():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, num_traces=51, nz_target=200)
    assert m["nx"] == 51 and m["x"].shape == (51,)
    assert m["width_m"] == pytest.approx(40.0)            # 400x200 image -> aspect 2
    assert m["dx"] == pytest.approx(40.0 / 50)
    assert m["dz"] == pytest.approx(0.1)                  # 20 m / 200 rows
    assert m["facies"].shape == m["vp"].shape == (m["nz"], 51)


def test_dz_floor_is_10cm():
    m = outcrop_to_model(_interp([BAND]), height_m=2.0, nz_target=400)
    assert m["dz"] == pytest.approx(0.1)


def test_band_rasterizes_at_expected_depth_and_pads_with_background():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, nz_target=200, pad_m=5.0)
    top = m["image_top_m"]
    assert top == pytest.approx(5.0)
    # rows inside the band (y 0.2-0.4 of 20 m -> 4-8 m below image top)
    z = m["z"]
    inside = (z > top + 4.05) & (z < top + 7.95)
    outside_above = z < top + 3.95
    assert np.all(m["facies"][inside, :] == 1)
    assert np.all(m["facies"][outside_above, :] == 0)
    assert np.all(m["facies"][:, 0] == m["facies"][:, -1])  # full-width band


def test_polygon_lens_is_laterally_bounded_and_later_region_wins():
    m = outcrop_to_model(_interp([BAND, LENS]), height_m=20.0, nz_target=200, pad_m=2.0)
    z, x = m["z"], m["x"]
    top = m["image_top_m"]
    zi = np.argmin(np.abs(z - (top + 0.75 * 20)))     # centre of lens in depth
    row = m["facies"][zi]
    assert row[np.argmin(np.abs(x - 0.5 * m["width_m"]))] == 2
    assert row[0] == 0 and row[-1] == 0
    # overlap test: a second band over the first -> later wins
    over = dict(BAND, id=7, label="silt", lithology="siltstone")
    m2 = outcrop_to_model(_interp([BAND, over]), height_m=20.0, nz_target=200, pad_m=2.0)
    assert 1 not in np.unique(m2["facies"]) and 7 in np.unique(m2["facies"])


def test_cover_is_background_and_not_in_grid():
    m = outcrop_to_model(_interp([COVER, BAND]), height_m=20.0)
    assert set(np.unique(m["facies"])) == {0, 1}
    prov = {r["id"]: r for r in m["regions"]}
    assert prov[3]["route"] == "background" and prov[3]["n_cells"] == 0


def test_elastic_grids_match_resolve_lithology():
    m = outcrop_to_model(_interp([BAND, LENS]), height_m=20.0)
    sand = resolve_lithology("sandstone")
    shale = resolve_lithology("shale")
    assert m["vp"][m["facies"] == 1].min() == pytest.approx(sand["vp"])
    assert m["vp"][m["facies"] == 0].max() == pytest.approx(shale["vp"])
    assert m["vp"][m["facies"] == 2].max() == pytest.approx(5000.0)
    assert np.all(m["vs"] < m["vp"]) and np.all(m["rho"] > 0)


def test_default_pad_is_1p5_wavelengths_of_background():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, wavelet_freq=25.0)
    v_bg = resolve_lithology("shale")["vp"]
    assert m["pad_m"] == pytest.approx(1.5 * v_bg / 25.0)
    assert m["z"][0] == pytest.approx(m["dz"] / 2)


def test_overrides_by_id_and_label():
    regs = _interp([BAND, LENS])["regions"]
    out = apply_overrides(regs, {"1": {"fluid": "gas"}, "lime lens": {"lithology": "dolomite"}})
    assert out[0]["fluid"] == "gas" and out[1]["lithology"] == "dolomite"
    assert "fluid" not in regs[0]                      # input untouched


def test_overrides_unknown_key_or_field_rejected():
    regs = _interp([BAND])["regions"]
    with pytest.raises(ValueError, match="no region"):
        apply_overrides(regs, {"99": {"fluid": "gas"}})
    with pytest.raises(ValueError, match="unknown override"):
        apply_overrides(regs, {"1": {"colour": "red"}})


def test_gas_override_changes_grid_and_direct_route_override_errors():
    brine = outcrop_to_model(_interp([BAND]), height_m=20.0)
    gas = outcrop_to_model(_interp([BAND]), height_m=20.0, overrides={1: {"fluid": "gas"}})
    # Robust Gassmann invariants (Vp alone is not one for stiff Han rocks): density and AI drop.
    assert gas["rho"][gas["facies"] == 1].max() < brine["rho"][brine["facies"] == 1].min()
    gas_ai = (gas["vp"] * gas["rho"])[gas["facies"] == 1].max()
    brine_ai = (brine["vp"] * brine["rho"])[brine["facies"] == 1].min()
    assert gas_ai < brine_ai
    assert gas["regions"][0]["fluid"] == "gas"
    with pytest.raises(ValueError, match="limestone"):
        outcrop_to_model(_interp([LENS]), height_m=20.0, overrides={2: {"fluid": "gas"}})


def test_background_override():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, background_lithology="limestone")
    assert m["legend"][0]["lithology"] == "limestone"
    assert m["vp"][m["facies"] == 0].max() == pytest.approx(5000.0)


def test_missing_image_size_warns_and_uses_default_aspect():
    d = _interp([BAND]); d.pop("image_size")
    with pytest.warns(UserWarning, match="aspect"):
        m = outcrop_to_model(d, height_m=20.0)
    assert m["width_m"] == pytest.approx(30.0)


def test_bad_geometry_params_rejected():
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=-5)
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=20, num_traces=1)
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=20, pad_m=0)
