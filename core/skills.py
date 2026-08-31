"""Reusable skills (Tier 4): YAML-defined, parameterized flows that run either
as a deterministic replay of a recorded tool chain or as an LLM-guided
procedure with a scoped toolset.

Pure half (this file's first part): model, validation, slot substitution,
parameterizer, two-layer registry, file IO. Execution (execute_skill) sits at
the bottom and is the only part that touches the loop/executor — imported
lazily to keep core.tool_registry free of import cycles.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_SLOT_RE = re.compile(r"\{\{(\w+)\}\}")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCALARS = (str, int, float, bool, type(None))
CONTEXT_PARAMS = ("image_path", "interpretation", "model")  # mirrors _CONTEXT_INPUTS


@dataclass
class Skill:
    name: str
    description: str
    parameters: Dict[str, dict]
    tools: List[str]
    procedure: str
    chain: List[dict] = field(default_factory=list)
    source: str = "memory"
    path: Optional[str] = None


@dataclass(frozen=True)
class SkillCard:
    """Duck-typed ToolSpec stand-in so ToolIndex.render_card works unchanged."""
    name: str
    description: str
    params: dict
    required: tuple
    auto_plot: Optional[str] = None


def _slots_in(value: Any) -> List[str]:
    if isinstance(value, str):
        return _SLOT_RE.findall(value)
    if isinstance(value, dict):
        return [s for v in value.values() for s in _slots_in(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _slots_in(v)]
    return []


def _registry_names() -> set:
    from core.tool_registry import REGISTRY_BY_NAME  # lazy: avoids import cycle
    return set(REGISTRY_BY_NAME)


def validate_skill(data: Any, source: str = "memory",
                   path: Optional[str] = None) -> Skill:
    """Turn a raw mapping into a Skill or raise ValueError naming the problem."""
    where = f" ({path})" if path else ""
    if not isinstance(data, dict):
        raise ValueError(f"skill{where}: expected a mapping")
    for key in ("name", "description", "parameters", "tools", "procedure"):
        if key not in data:
            raise ValueError(f"skill{where}: missing required key '{key}'")
    name = data["name"]
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"skill{where}: invalid name {name!r} (use [a-z][a-z0-9_]*)")
    parameters = data["parameters"] or {}
    if not isinstance(parameters, dict) or not all(isinstance(v, dict) for v in parameters.values()):
        raise ValueError(f"skill {name}: parameters must map names to schema dicts")
    tools = list(data["tools"] or [])
    known = _registry_names()
    for t in tools:
        if t not in known:
            raise ValueError(f"skill {name}: unknown tool '{t}'")
    chain = list(data.get("chain") or [])
    for i, step in enumerate(chain):
        if not isinstance(step, dict) or "tool" not in step:
            raise ValueError(f"skill {name}: chain step {i} must have a 'tool'")
        if step["tool"] not in known:
            raise ValueError(f"skill {name}: chain step {i} unknown tool '{step['tool']}'")
        if step["tool"] not in tools:
            raise ValueError(f"skill {name}: chain tool '{step['tool']}' not in tools")
        if not isinstance(step.get("args", {}), dict):
            raise ValueError(f"skill {name}: chain step {i} args must be a mapping")
    used = set(_slots_in(data["procedure"])) | set(_slots_in([s.get("args", {}) for s in chain]))
    undeclared = sorted(used - set(parameters))
    if undeclared:
        raise ValueError(f"skill {name}: undeclared slot(s) {', '.join(undeclared)}")
    return Skill(name=name, description=str(data["description"]), parameters=parameters,
                 tools=tools, procedure=str(data["procedure"]),
                 chain=[{"tool": s["tool"], "args": dict(s.get("args", {}))} for s in chain],
                 source=source, path=path)


def substitute(value: Any, params: Dict[str, Any]) -> Any:
    """Value-level templating: an exact '{{slot}}' string becomes the typed
    parameter value; slots inside longer strings are replaced textually."""
    if isinstance(value, str):
        m = _SLOT_RE.fullmatch(value)
        if m:
            return params[m.group(1)]
        return _SLOT_RE.sub(lambda mm: str(params[mm.group(1)]), value)
    if isinstance(value, dict):
        return {k: substitute(v, params) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [substitute(v, params) for v in value]
    return value


def fill_procedure(procedure: str, params: Dict[str, Any]) -> str:
    return _SLOT_RE.sub(lambda m: str(params[m.group(1)]), procedure)


def resolve_params(skill: Skill, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params or {})
    unknown = sorted(set(params) - set(skill.parameters))
    if unknown:
        raise ValueError(f"skill {skill.name}: unknown parameter(s) {', '.join(unknown)}")
    bound: Dict[str, Any] = {}
    for pname, schema in skill.parameters.items():
        if pname in params:
            bound[pname] = params[pname]
        elif "default" in schema:
            bound[pname] = schema["default"]
        else:
            raise ValueError(f"skill {skill.name}: missing required parameter '{pname}'")
    return bound


# --- capture -----------------------------------------------------------------

def _is_portable(value: Any, max_list: int) -> bool:
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) <= max_list and all(isinstance(v, _SCALARS) for v in value)
    return False


def _match_slot(value: Any, param_values: Dict[str, Any]) -> Optional[str]:
    for slot, pv in param_values.items():
        if isinstance(value, bool) or isinstance(pv, bool):
            if value is pv:
                return slot
            continue
        if isinstance(value, (int, float)) and isinstance(pv, (int, float)):
            if float(value) == float(pv):
                return slot
        elif isinstance(value, str) and isinstance(pv, str) and value == pv:
            return slot
    return None


def build_chain(calls: Iterable[dict], param_values: Dict[str, Any],
                context_params: set, max_list: int = 12) -> Tuple[List[str], List[dict]]:
    tools: List[str] = []
    chain: List[dict] = []
    used: set = set()
    for call in calls:
        args: Dict[str, Any] = {}
        for k, v in (call.get("args") or {}).items():
            if k in context_params or k == "_session" or not _is_portable(v, max_list):
                continue
            slot = _match_slot(v, param_values)
            if slot is not None:
                args[k] = "{{" + slot + "}}"
                used.add(slot)
            else:
                args[k] = v
        chain.append({"tool": call["tool"], "args": args})
        if call["tool"] not in tools:
            tools.append(call["tool"])
    for slot, pv in param_values.items():
        if slot not in used:
            raise ValueError(f"parameter {slot}={pv!r} was not used by any tool call")
    return tools, chain


def _value_forms(value: Any) -> List[str]:
    forms = [str(value)]
    if isinstance(value, float) and value.is_integer():
        forms.append(str(int(value)))
    if isinstance(value, int) and not isinstance(value, bool):
        forms.append(f"{value}.0")
    return forms


def build_procedure(input_text: str, param_values: Dict[str, Any],
                    tools: List[str]) -> str:
    text = (input_text or "").strip()
    if not text:
        return "Run the recorded chain: " + " → ".join(tools) + "."
    for slot, pv in sorted(param_values.items(), key=lambda kv: -len(str(kv[1]))):
        for form in _value_forms(pv):
            text = re.sub(rf"(?<![\w.]){re.escape(form)}(?![\w.])", "{{" + slot + "}}", text)
    return text


def _normalize_parameters(parameters: Dict[str, Any]) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """Accept {slot: value} or {slot: {value, description?, type?}}."""
    schemas: Dict[str, dict] = {}
    values: Dict[str, Any] = {}
    for slot, spec in (parameters or {}).items():
        if not _NAME_RE.match(str(slot)):
            raise ValueError(f"invalid parameter name {slot!r}")
        if isinstance(spec, dict) and "value" in spec:
            value = spec["value"]
            ptype = spec.get("type") or ("number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string")
            schemas[slot] = {"type": ptype, "description": spec.get("description", slot), "default": value}
        else:
            value = spec
            ptype = "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string"
            schemas[slot] = {"type": ptype, "description": slot, "default": value}
        values[slot] = value
    if not values:
        raise ValueError("at least one parameter is required to make a skill reusable")
    return schemas, values


def capture_skill(name: str, description: str, parameters: Dict[str, Any],
                  calls: List[dict], input_text: str, context_params: set) -> dict:
    if not calls:
        raise ValueError("the last turn ran no tools — nothing to capture")
    schemas, values = _normalize_parameters(parameters)
    tools, chain = build_chain(calls, values, context_params)
    data = {"name": name, "description": description, "parameters": schemas,
            "tools": tools, "procedure": build_procedure(input_text, values, tools),
            "chain": chain}
    validate_skill(data)
    return data


# --- registry ----------------------------------------------------------------

class SkillRegistry:
    def __init__(self, repo_dir: Optional[str] = None, runtime_dir: Optional[str] = None):
        if repo_dir is None or runtime_dir is None:
            from config.settings import SEISMIC_SKILLS_DIR, SKILLS_REPO_DIR
            repo_dir = repo_dir or SKILLS_REPO_DIR
            runtime_dir = runtime_dir or SEISMIC_SKILLS_DIR
        self.repo_dir = repo_dir
        self.runtime_dir = runtime_dir
        self._skills: Dict[str, Skill] = {}
        self.reload()

    def _load_dir(self, directory: str, source: str) -> None:
        if not directory or not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, fname)
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                skill = validate_skill(data, source=source, path=path)
            except Exception as e:
                logger.warning(f"skipping skill file {path}: {e}")
                continue
            if skill.name in self._skills and source == "runtime":
                logger.warning(f"runtime skill '{skill.name}' overrides the repo skill")
            self._skills[skill.name] = skill

    def reload(self) -> None:
        self._skills = {}
        self._load_dir(self.repo_dir, "repo")
        self._load_dir(self.runtime_dir, "runtime")

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise ValueError(f"unknown skill '{name}'; use list_skills to see available skills")
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills)

    def list(self) -> List[dict]:
        return [{"name": s.name, "description": s.description, "parameters": s.parameters,
                 "has_chain": bool(s.chain), "source": s.source}
                for s in (self._skills[n] for n in self.names())]

    def specs(self) -> List[SkillCard]:
        return [SkillCard(name=f"skill:{s.name}", description=s.description,
                          params=s.parameters,
                          required=tuple(p for p, sch in s.parameters.items() if "default" not in sch))
                for s in (self._skills[n] for n in self.names())]

    def save(self, data: dict, overwrite: bool = False) -> str:
        skill = validate_skill(data, source="runtime")
        if skill.name in _registry_names():
            raise ValueError(f"skill name '{skill.name}' collides with a registry tool")
        os.makedirs(self.runtime_dir, mode=0o700, exist_ok=True)
        path = os.path.join(self.runtime_dir, f"{skill.name}.yaml")
        if os.path.exists(path) and not overwrite:
            raise ValueError(f"skill '{skill.name}' already exists at {path}; pass overwrite=true to replace")
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        self.reload()
        return path


_REGISTRY: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY


def set_registry(registry: Optional[SkillRegistry]) -> None:
    global _REGISTRY
    _REGISTRY = registry


# --- execution -----------------------------------------------------------------

def execute_skill(skill: Skill, params: Optional[Dict[str, Any]], mode: str,
                  session: Any) -> Dict[str, Any]:
    """Run a skill: deterministic replay of its chain through the session's
    ToolLoopRunner.execute_call, or LLM-guided via a scoped ExecutorAgent."""
    from core.turn_trace import emit_event  # lazy: keep this module import-light
    if session is None:
        raise ValueError("run_skill requires a live session (call it from the chat loop)")
    cm = session.context_manager
    if (cm.get_context("_skill_depth") or 0) >= 1:
        raise ValueError("run_skill cannot be invoked from inside a running skill")
    if mode not in ("auto", "replay", "guided"):
        raise ValueError(f"mode must be auto, replay or guided (got {mode!r})")
    if mode == "replay" and not skill.chain:
        raise ValueError(f"skill '{skill.name}' has no recorded chain; use mode='guided'")
    bound = resolve_params(skill, params or {})
    use_replay = mode == "replay" or (mode == "auto" and bool(skill.chain))
    runner = session.runner
    cm.set_context("_skill_depth", 1)
    runner.current_skill = skill.name
    try:
        if use_replay:
            images: List[str] = []
            steps: List[dict] = []
            last: Any = None
            for step in skill.chain:
                args = substitute(step["args"], bound)
                try:
                    last = runner.execute_call(step["tool"], args, images)
                    steps.append({"tool": step["tool"], "ok": True})
                except Exception as e:
                    steps.append({"tool": step["tool"], "ok": False, "error": str(e)})
                    emit_event(cm, "tool_call", tool=step["tool"], ok=False, error=str(e))
                    emit_event(cm, "skill_run", name=skill.name, mode="replay",
                               n_steps=len(steps), error=str(e))
                    return {"mode": "replay", "steps": steps,
                            "error": f"step {step['tool']} failed: {e}",
                            "extra_image_paths": images}
            emit_event(cm, "skill_run", name=skill.name, mode="replay", n_steps=len(steps))
            return {"mode": "replay", "steps": steps,
                    "result": runner.compact_value(last), "extra_image_paths": images}
        from core.executor_agent import ExecutorAgent
        brief = fill_procedure(skill.procedure, bound)
        result = ExecutorAgent(session.llm_client, session.tool_manager, cm).run(
            brief, list(skill.tools))
        emit_event(cm, "skill_run", name=skill.name, mode="guided",
                   n_steps=len(result.tools_used))
        out: Dict[str, Any] = {"mode": "guided", "summary": result.summary,
                               "tools_used": list(result.tools_used),
                               "extra_image_paths": list(result.images)}
        if result.error:
            out["error"] = result.error
        return out
    finally:
        runner.current_skill = None
        cm.set_context("_skill_depth", 0)
