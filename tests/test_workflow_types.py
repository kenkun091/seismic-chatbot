import pytest

from workflows.types import Layer, Scenario


def test_layer_fields_and_default_label():
    ly = Layer(vp=3000.0, vs=1500.0, rho=2.2)
    assert (ly.vp, ly.vs, ly.rho) == (3000.0, 1500.0, 2.2)
    assert ly.label == ""


def test_layer_is_frozen():
    ly = Layer(vp=3000.0, vs=1500.0, rho=2.2)
    with pytest.raises(Exception):
        ly.vp = 9999.0  # frozen dataclass forbids reassignment


def test_scenario_holds_named_layers():
    brine = Layer(3000.0, 1500.0, 2.20, "brine")
    gas = Layer(2700.0, 1550.0, 2.05, "gas")
    sc = Scenario(name="fluid", cases={"brine": brine, "gas": gas})
    assert sc.name == "fluid"
    assert sc.cases["gas"].vp < sc.cases["brine"].vp
