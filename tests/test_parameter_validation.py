from tools.parameter_validation import validate_make_ricker, validate_wedge_model, validate_avo


def test_ricker_ok():
    ok, msg = validate_make_ricker({"frequency": 30, "dt": 0.001})
    assert ok and msg == ""


def test_ricker_bad_frequency():
    ok, msg = validate_make_ricker({"frequency": 0})
    assert not ok and "Frequency" in msg


def test_ricker_bad_dt():
    ok, msg = validate_make_ricker({"frequency": 30, "dt": 0.5})
    assert not ok and "dt" in msg.lower()


def test_wedge_ok():
    ok, msg = validate_wedge_model({
        "max_thickness": 50, "v1": 2500, "v2": 3000, "v3": 3500,
        "rho1": 2.2, "rho2": 2.3, "rho3": 2.4,
    })
    assert ok and msg == ""


def test_wedge_bad_velocity():
    # validate_wedge_model enforces positivity only (not ordering/range);
    # v1=0 is non-positive and must be rejected
    ok, msg = validate_wedge_model({
        "max_thickness": 50, "v1": 0, "v2": 3000, "v3": 3500,
        "rho1": 2.2, "rho2": 2.3, "rho3": 2.4,
    })
    assert not ok and "v1" in msg


def test_wedge_bad_thickness():
    ok, msg = validate_wedge_model({
        "max_thickness": -5, "v1": 2500, "v2": 3000, "v3": 3500,
        "rho1": 2.2, "rho2": 2.3, "rho3": 2.4,
    })
    assert not ok and "thickness" in msg.lower()


def test_avo_missing_param():
    ok, msg = validate_avo({"vp1": 2500})
    assert not ok and "Missing" in msg
