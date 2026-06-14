import os

import numpy as np
import pytest

from tools.wedge_tools import wedge_avo_gather, create_wedge_model

GKW = dict(max_thickness=60, v1=2500, v2=3000, v3=3500, rho1=2.2, rho2=2.3, rho3=2.4)


def test_gather_shape():
    angles = [0, 10, 20, 30]
    t, cube, params = wedge_avo_gather(angles=angles, **GKW)
    assert cube.ndim == 3
    assert cube.shape == (len(t), 61, len(angles))
    assert params["num_traces"] == 61
    assert params["angles"] == angles


def test_single_angle_panel_matches_wedge_model():
    # At a non-zero angle both paths use Shuey, so the gather panel must equal the
    # single-angle wedge_model synthetic exactly (same geometry + same RC).
    ang = 10
    _, cube, _ = wedge_avo_gather(angles=[ang], **GKW)
    _, _, synth, _ = create_wedge_model(incident_angle=ang, **GKW)
    synth = np.asarray(synth)
    assert cube.shape[:2] == synth.shape
    assert np.allclose(cube[:, :, 0], synth, atol=1e-9)


def test_gather_accepts_velocity_inversion():
    _, cube, _ = wedge_avo_gather(
        angles=[10, 20], max_thickness=50,
        v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4,
    )
    assert cube.shape[2] == 2
    assert np.all(np.isfinite(cube))


def test_gather_rejects_vs_ge_vp():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[10], vs1=3000, **GKW)  # vs1>=vp1=2500


def test_gather_rejects_vs_equal_vp():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[10], vs1=2500, **GKW)  # vs1 == vp1 (boundary)


def test_gather_rejects_bad_angle():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[95], **GKW)


def test_gather_rejects_empty_angles():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[], **GKW)


from tools.wedge_tools import analyze_wedge_gather


def test_analyze_gather_tuning_and_avo_keys():
    angles = [0, 15, 30]
    _, cube, params = wedge_avo_gather(angles=angles, **GKW)  # v2=3000, f=30 -> tuning ~25 m
    out = analyze_wedge_gather(cube, params)
    assert abs(out["tuning_thickness"] - 25.0) < 1e-6
    assert len(out["per_angle"]) == 3
    assert out["per_angle"][0]["angle"] == 0
    assert set(out["avo"].keys()) == {"angles", "amplitudes"}
    assert len(out["avo"]["amplitudes"]) == 3


def test_analyze_gather_avo_varies_with_angle():
    # Gas-sand contrast -> AVO amplitude must vary across angles (not constant).
    angles = [0, 15, 30, 40]
    _, cube, params = wedge_avo_gather(
        angles=angles, max_thickness=60,
        v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4)
    out = analyze_wedge_gather(cube, params)
    amps = out["avo"]["amplitudes"]
    assert max(amps) - min(amps) > 1e-6
