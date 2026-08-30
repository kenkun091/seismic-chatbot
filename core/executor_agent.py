"""Executor subagent: one task brief, a scoped toolset, the shared tool loop."""
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.tool_loop import ToolLoopRunner
from core.tool_registry import REGISTRY_BY_NAME, to_openai_schema
from core.tool_index import render_card

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """You are a seismic modeling task executor. Complete ONE task using only your assigned tools.

Rules:
- Tool results are compacted before you see them: long numeric arrays appear as summaries like "<61 values, min=..., max=...>".
- Any plot a tool produces is displayed to the user automatically — never print or mention image file paths.
- Plot tools run automatically after their matching compute tool — never call a plot_* tool yourself, and never pass raw array data as tool arguments.
- Never pass image_path, interpretation or model arguments yourself — they are supplied automatically from session context.
- When done, state the key quantitative results (e.g. tuning thickness, AVO class, intercept/gradient) inside <reply></reply> XML tags.

Your assigned tools:
{cards}
"""


@dataclass
class TaskResult:
    summary: str
    images: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    error: Optional[str] = None


class ExecutorAgent:
    def __init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5):
        self._loop = ToolLoopRunner(llm_client, tool_manager, context_manager,
                                    max_tool_rounds=max_tool_rounds)

    def run(self, brief: str, tool_names: list) -> TaskResult:
        unknown = [n for n in tool_names if n not in REGISTRY_BY_NAME]
        if unknown:
            return TaskResult(summary="", error=f"Unknown tool(s): {', '.join(unknown)}")
        specs = [REGISTRY_BY_NAME[n] for n in tool_names]
        schemas = [{"type": "function", "function": to_openai_schema(s)} for s in specs]
        system_prompt = EXECUTOR_SYSTEM_PROMPT.format(
            cards="\n".join(f"- {render_card(s)}" for s in specs))
        try:
            out = self._loop.run(system_prompt,
                                 [{"role": "user", "content": brief}], schemas)
        except Exception as e:  # a failure here must not kill the orchestrator turn
            logger.error(f"Executor failed on brief {brief!r}: {e}", exc_info=True)
            return TaskResult(summary="", error=str(e))
        return TaskResult(summary=out["reply"], images=out["images"],
                          tools_used=out["tools_used"])
