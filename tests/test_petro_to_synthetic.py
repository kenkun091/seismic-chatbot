"""petro_to_synthetic recipe: per-layer petrophysics -> N-layer synthetic."""
import json
import os

import pytest

from workflows.recipes.petro_to_synthetic import petro_to_synthetic

PHIT = [0.10, 0.25, 0.10]
VCLAY = [0.55, 0.10, 0.55]
TH = [30.0, 20.0]


def _cleanup(result):
    if os.path.exists(result.get("image_path", "")):
        os.remove(result["image_path"])


def test_end_to_end_brine_stack():
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        assert res["n_layers"] == 3
        assert len(res["layers"]) == 3
        assert len(res["interface_times"]) == 2
        assert len(res["rcs"]) == 2
        assert res["max_abs_amplitude"] > 0
        assert res["max_abs_rc"] > 0
        assert all(ly["fluid"] == "brine" for ly in res["layers"])
        assert res["layers"][0]["label"] == "layer 1"
        assert os.path.getsize(res["image_path"]) > 0
    finally:
        _cleanup(res)


def test_layers_match_predict_layer():
    from workflows.adapters import predict_layer
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        expected = predict_layer(PHIT[1], VCLAY[1], fluid="brine", label="layer 2")
        assert res["layers"][1]["vp"] == pytest.approx(expected.vp)
        assert res["layers"][1]["vs"] == pytest.approx(expected.vs)
        assert res["layers"][1]["rho"] == pytest.approx(expected.rho)
    finally:
        _cleanup(res)


def test_gas_layer_lowers_vp_and_raises_vs():
    brine = petro_to_synthetic(PHIT, VCLAY, TH)
    gas = petro_to_synthetic(PHIT, VCLAY, TH, fluids=["brine", "gas", "brine"])
    try:
        assert gas["layers"][1]["vp"] < brine["layers"][1]["vp"]
        assert gas["layers"][1]["vs"] > brine["layers"][1]["vs"]  # Gassmann: mu fluid-independent, rho drops
    finally:
        _cleanup(brine)
        _cleanup(gas)


def test_custom_labels_flow_through():
    res = petro_to_synthetic(PHIT, VCLAY, TH, labels=["shale", "sand", "shale"])
    try:
        assert [ly["label"] for ly in res["layers"]] == ["shale", "sand", "shale"]
    finally:
        _cleanup(res)


def test_result_is_json_serializable():
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        json.dumps(res)
    finally:
        _cleanup(res)


class TestRecipeGuards:
    def test_fewer_than_two_layers(self):
        with pytest.raises(ValueError, match="at least 2 layers"):
            petro_to_synthetic([0.2], [0.1], [])

    def test_vclay_length_mismatch(self):
        with pytest.raises(ValueError, match="vclay must have 3"):
            petro_to_synthetic(PHIT, [0.5, 0.1], TH)

    def test_thickness_length_rule(self):
        with pytest.raises(ValueError, match=r"len\(phit\)-1 = 2"):
            petro_to_synthetic(PHIT, VCLAY, [30.0])

    def test_fluids_length_mismatch(self):
        with pytest.raises(ValueError, match="fluids must have 3"):
            petro_to_synthetic(PHIT, VCLAY, TH, fluids=["brine"])

    def test_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="labels must have 3"):
            petro_to_synthetic(PHIT, VCLAY, TH, labels=["a"])

    def test_non_positive_thickness(self):
        with pytest.raises(ValueError, match=r"thickness\[0\]"):
            petro_to_synthetic(PHIT, VCLAY, [-5.0, 20.0])


class TestWorkflowRegistration:
    def test_in_workflow_and_tool_registries(self):
        from workflows.engine import WORKFLOW_NAMES, WORKFLOW_REGISTRY_BY_NAME
        from core.tool_registry import TOOL_FUNCTIONS
        assert "petro_to_synthetic" in WORKFLOW_NAMES
        assert "petro_to_synthetic" in TOOL_FUNCTIONS
        spec = WORKFLOW_REGISTRY_BY_NAME["petro_to_synthetic"]
        assert spec.required == ["phit", "vclay", "thickness"]

    def test_engine_run_fills_defaults(self):
        from workflows.engine import WorkflowEngine
        res = WorkflowEngine().run("petro_to_synthetic",
                                   {"phit": PHIT, "vclay": VCLAY, "thickness": TH})
        try:
            assert res["wavelet_freq"] == 30.0 and res["angle"] == 0.0
        finally:
            _cleanup(res)

    def test_system_prompt_lists_recipe(self, fake_llm_factory):
        from core.chatbot_tool_use import SeismicChatBotToolUse
        bot = SeismicChatBotToolUse(llm_client=fake_llm_factory([]))
        assert "- petro_to_synthetic:" in bot._create_system_prompt()

    def test_run_sweep_over_wavelet_freq(self):
        from workflows.sweep import run_sweep
        res = run_sweep(
            "petro_to_synthetic",
            grid={"wavelet_freq": [20.0, 40.0]},
            metric="max_abs_amplitude",
            fixed={"phit": PHIT, "vclay": VCLAY, "thickness": TH},
        )
        try:
            assert res["coverage"] == {"total": 2, "ran": 2, "failed": 0,
                                       "failures": []}
            assert res["stats"]["kind"] == "numeric"
            assert len(res["rows"]) == 2
        finally:
            _cleanup(res)
