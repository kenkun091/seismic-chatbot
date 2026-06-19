"""Typed data spine for workflow recipes.

A `Layer` is one rock (Vp, Vs, density); a `Scenario` bundles named layers
(e.g. a brine case vs a gas case). These live inside the workflow engine —
leaf tools never see them; adapters translate Layer <-> the {vp1, vs1, ...}
dicts the leaf tools expect.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Layer:
    """One elastic rock layer. Units: vp, vs in m/s; rho in g/cm^3."""
    vp: float
    vs: float
    rho: float
    label: str = ""


@dataclass(frozen=True)
class Scenario:
    """A named bundle of layers, e.g. Scenario("fluid", {"brine": ..., "gas": ...})."""
    name: str
    cases: dict = field(default_factory=dict)
