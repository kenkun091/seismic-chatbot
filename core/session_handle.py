"""Handle injected into session-scoped registry tools (run_skill, save_skill,
list_skills) as the hidden `_session` kwarg — never in LLM-facing schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SessionHandle:
    llm_client: Any
    tool_manager: Any
    context_manager: Any
    runner: Any  # the ToolLoopRunner executing the current call
