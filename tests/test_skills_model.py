import pytest
import yaml

from core.skills import (CONTEXT_PARAMS, SkillRegistry, build_chain, build_procedure,
                         capture_skill, fill_procedure, resolve_params, substitute,
                         validate_skill)

_GOOD = {
    "name": "ricker_wavelet",
    "description": "Build a Ricker wavelet.",
    "parameters": {"frequency": {"type": "number", "description": "Hz", "default": 30}},
    "tools": ["make_ricker"],
    "procedure": "Create a {{frequency}} Hz Ricker wavelet.",
    "chain": [{"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}],
}


def test_validate_good_skill():
    s = validate_skill(_GOOD)
    assert s.name == "ricker_wavelet" and s.chain[0]["tool"] == "make_ricker"


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("procedure"), "procedure"),
    (lambda d: d.update(name="Bad Name"), "name"),
    (lambda d: d.update(tools=["no_such_tool"]), "no_such_tool"),
    (lambda d: d.update(chain=[{"tool": "wedge_model", "args": {}}]), "wedge_model"),
    (lambda d: d.update(procedure="use {{nope}}"), "nope"),
    (lambda d: d.update(chain=[{"tool": "make_ricker", "args": {"frequency": "{{nope}}"}}]), "nope"),
])
def test_validate_rejects(mutate, msg):
    data = yaml.safe_load(yaml.safe_dump(_GOOD))
    mutate(data)
    with pytest.raises(ValueError) as exc:
        validate_skill(data)
    assert msg in str(exc.value)


def test_substitute_typed_and_textual():
    params = {"freq": 30, "name": "sand"}
    assert substitute("{{freq}}", params) == 30            # exact slot -> typed value
    assert substitute("{{freq}} Hz {{name}}", params) == "30 Hz sand"
    assert substitute({"a": ["{{freq}}", "x"]}, params) == {"a": [30, "x"]}
    assert substitute(5, params) == 5


def test_resolve_params_defaults_unknown_required():
    s = validate_skill(_GOOD)
    assert resolve_params(s, {}) == {"frequency": 30}
    assert resolve_params(s, {"frequency": 45}) == {"frequency": 45}
    with pytest.raises(ValueError):
        resolve_params(s, {"bogus": 1})
    strict = dict(_GOOD, parameters={"frequency": {"type": "number"}})
    with pytest.raises(ValueError):
        resolve_params(validate_skill(strict), {})


def test_build_chain_parameterizes_by_value_and_drops_context_args():
    calls = [
        {"tool": "make_ricker", "args": {"frequency": 30, "time_length": 200}, "ok": True},
        {"tool": "interpret_outcrop", "args": {"image_path": "/tmp/x.png"}, "ok": True},
        {"tool": "wedge_model", "args": {"wavelet_freq": 30.0, "v1": 2500,
                                        "big": list(range(50))}, "ok": True},
    ]
    tools, chain = build_chain(calls, {"freq": 30}, set(CONTEXT_PARAMS))
    assert tools == ["make_ricker", "interpret_outcrop", "wedge_model"]
    assert chain[0]["args"] == {"frequency": "{{freq}}", "time_length": 200}
    assert chain[1]["args"] == {}                      # context arg dropped
    assert chain[2]["args"] == {"wavelet_freq": "{{freq}}", "v1": 2500}  # 30.0 == 30; big list dropped


def test_build_chain_rejects_unused_parameter():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    with pytest.raises(ValueError) as exc:
        build_chain(calls, {"thickness": 100}, set(CONTEXT_PARAMS))
    assert "thickness=100" in str(exc.value)


def test_build_procedure_substitutes_values_or_falls_back():
    assert build_procedure("make a 30 Hz ricker with 0.25 porosity",
                           {"freq": 30, "phit": 0.25}, ["make_ricker"]) == \
        "make a {{freq}} Hz ricker with {{phit}} porosity"
    assert build_procedure("", {"freq": 30}, ["make_ricker", "analyze_wedge"]) == \
        "Run the recorded chain: make_ricker → analyze_wedge."


def test_capture_skill_produces_valid_skill():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    data = capture_skill("my_ricker", "A ricker.", {"freq": 30}, calls,
                         "make a 30 Hz ricker", set(CONTEXT_PARAMS))
    s = validate_skill(data)
    assert s.tools == ["make_ricker"]
    assert s.parameters["freq"]["default"] == 30
    assert s.chain == [{"tool": "make_ricker", "args": {"frequency": "{{freq}}"}}]
    assert fill_procedure(s.procedure, {"freq": 45}) == "make a 45 Hz ricker"


def test_capture_accepts_rich_parameter_form():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    data = capture_skill("r", "d", {"freq": {"value": 30, "description": "Hz"}}, calls,
                         "x", set(CONTEXT_PARAMS))
    assert data["parameters"]["freq"] == {"type": "number", "description": "Hz", "default": 30}


def test_registry_two_layers_override_and_save(tmp_path, caplog):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "ricker_wavelet.yaml").write_text(yaml.safe_dump(_GOOD, sort_keys=False))
    reg = SkillRegistry(repo_dir=str(repo), runtime_dir=str(runtime))
    assert reg.names() == ["ricker_wavelet"]
    assert reg.get("ricker_wavelet").source == "repo"
    override = dict(_GOOD, description="runtime version")
    path = reg.save(override)
    assert path == str(runtime / "ricker_wavelet.yaml")
    assert reg.get("ricker_wavelet").description == "runtime version"
    assert reg.get("ricker_wavelet").source == "runtime"
    assert any("overrides" in r.message for r in caplog.records)
    with pytest.raises(ValueError):
        reg.save(override)  # exists, overwrite=False
    reg.save(override, overwrite=True)
    with pytest.raises(ValueError):
        reg.save(dict(_GOOD, name="make_ricker"))  # collides with a registry tool
    cards = reg.specs()
    assert cards[0].name == "skill:ricker_wavelet" and cards[0].required == ()
    assert reg.list()[0]["has_chain"] is True


def test_registry_skips_invalid_files_with_warning(tmp_path, caplog):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "broken.yaml").write_text("name: broken\n")
    reg = SkillRegistry(repo_dir=str(tmp_path / "none"), runtime_dir=str(runtime))
    assert reg.names() == []
    assert any("broken" in r.message for r in caplog.records)


def test_settings_expose_skill_dirs():
    from config.settings import SEISMIC_SKILLS_DIR, SKILLS_REPO_DIR
    assert SEISMIC_SKILLS_DIR and SKILLS_REPO_DIR.endswith("skills")
