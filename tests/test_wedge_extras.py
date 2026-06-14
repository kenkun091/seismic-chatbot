import os
import numpy as np
import pytest
from tools.wedge_tools import parse_and_prep_wavelet, analyze_wedge, create_wedge_model


def test_parse_wavelet_from_string():
    t, w = parse_and_prep_wavelet("0,0.5,1,0.5,0,-0.5", dt=0.001)
    assert len(w) == 6
    assert np.isfinite(w).all()
    assert len(t) == len(w)


def test_parse_wavelet_from_array():
    t, w = parse_and_prep_wavelet([0.0, 1.0, 0.0], dt=0.001)
    assert list(w) == [0.0, 1.0, 0.0]


def test_parse_wavelet_from_file(tmp_path):
    p = tmp_path / "wav.txt"
    p.write_text("0\n0.5\n1\n0.5\n0\n")
    t, w = parse_and_prep_wavelet(str(p), dt=0.001)
    assert len(w) == 5
    assert w.max() == 1.0


def test_parse_wavelet_malformed_raises():
    with pytest.raises(ValueError):
        parse_and_prep_wavelet("not,a,number,x", dt=0.001)


def test_analyze_wedge_tuning_thickness():
    _, _, synthetic, params = create_wedge_model(
        max_thickness=60, v1=2500, v2=3000, v3=3500,
        rho1=2.2, rho2=2.3, rho3=2.4, wavelet_freq=30, num_traces=41,
    )
    out = analyze_wedge(synthetic_data=synthetic, parameters=params)
    # tuning thickness = v2 / (4 * freq) = 3000 / 120 = 25.0 m
    assert abs(out["tuning_thickness"] - 25.0) < 1e-6
    assert "max_amplitudes" in out


def test_csv_export_writes_file(tmp_path):
    out_csv = tmp_path / "curves.csv"
    create_wedge_model(
        max_thickness=60, v1=2500, v2=3000, v3=3500,
        rho1=2.2, rho2=2.3, rho3=2.4, wavelet_freq=30, num_traces=41,
        export_path=str(out_csv),
    )
    assert out_csv.exists()
    header = out_csv.read_text().splitlines()[0]
    assert "Thickness" in header
