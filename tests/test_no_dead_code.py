import importlib
import pytest


def test_calculate_AnB_gone():
    import tools.avo_tools as avo
    assert not hasattr(avo, "calculate_AnB")


def test_interactive_plotting_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tools.interactive_plotting")
