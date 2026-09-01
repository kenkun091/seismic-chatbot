"""LLM-facing skill tools (session-scoped: the loop injects `_session`).
All core.skills imports are lazy so core.tool_registry can import this module
without a cycle."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_skill(name: str, params: Optional[Dict[str, Any]] = None, mode: str = "auto",
              _session: Any = None) -> Dict[str, Any]:
    from core.skills import execute_skill, get_registry
    return execute_skill(get_registry().get(name), params or {}, mode, _session)


def save_skill(name: str, description: str, parameters: Dict[str, Any],
               overwrite: bool = False, _session: Any = None) -> Dict[str, Any]:
    from core.skills import CONTEXT_PARAMS, capture_skill, get_registry
    if _session is None:
        raise ValueError("save_skill requires a live session")
    cm = _session.context_manager
    calls = cm.get_context("last_turn_calls") or []
    data = capture_skill(name, description, parameters, calls,
                         cm.get_context("last_turn_input") or "", set(CONTEXT_PARAMS))
    registry = get_registry()
    path = registry.save(data, overwrite=bool(overwrite))
    _refresh_index(_session, registry)
    return {"name": name, "path": path, "n_steps": len(data["chain"]),
            "parameters": sorted(data["parameters"])}


def list_skills(_session: Any = None) -> List[dict]:
    from core.skills import get_registry
    return get_registry().list()


def _refresh_index(session: Any, registry: Any) -> None:
    """Re-index discovery if the session's runner exposes a tool index."""
    index = getattr(session, "tool_index", None)
    if index is not None and hasattr(index, "refresh"):
        try:
            index.refresh(registry.specs())
        except Exception:  # discovery is best-effort; the skill is saved regardless
            pass
