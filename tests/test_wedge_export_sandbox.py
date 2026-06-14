import pytest

from tools.wedge_tools import create_wedge_model

WEDGE_KW = dict(
    max_thickness=50.0, v1=2500, v2=2600, v3=2700, rho1=2.3, rho2=2.4, rho3=2.5,
)


def test_traversal_export_path_rejected():
    """create_wedge_model must reject a path-traversal export_path before doing work."""
    with pytest.raises(ValueError):
        create_wedge_model(export_path="../../evil.csv", **WEDGE_KW)


def test_absolute_export_path_rejected():
    with pytest.raises(ValueError):
        create_wedge_model(export_path="/tmp/evil.csv", **WEDGE_KW)
