"""Agentic-mode chatbot: a meta-tool loop that discovers tools semantically and
delegates domain tasks to scoped ExecutorAgents.

The LLM in this loop NEVER sees real tool schemas — only discover_tools and
run_task. The system prompt must not grow with the registry."""
import json
import logging
import uuid
from typing import Any, Dict, List

from core.llm_client import LLMClient
from core.tool_manager import ToolManager
from core.context_manager import ContextManager
from core.tool_registry import REGISTRY_BY_NAME
from core.tool_loop import extract_reply
from core.tool_index import ToolIndex
from core.turn_trace import emit_event, usage_dict
from core.executor_agent import ExecutorAgent
from core.knowledge_router import KnowledgeRouter
from knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

MAX_ORCH_ROUNDS = 8
META_TOOL_NAMES = ("discover_tools", "run_task")

META_TOOLS = [
    {"type": "function", "function": {
        "name": "discover_tools",
        "description": ("Semantic search over the seismic tool catalog. Describe ONE task "
                        "in plain words; returns the most relevant tools with their "
                        "required parameters. Call this before run_task."),
        "parameters": {"type": "object", "properties": {
            "task_description": {"type": "string",
                                 "description": "Plain-language description of one task."}},
            "required": ["task_description"]}}},
    {"type": "function", "function": {
        "name": "run_task",
        "description": ("Delegate one task to an executor subagent. Give a self-contained "
                        "brief (include every numeric parameter the user supplied) and the "
                        "tool names chosen from discover_tools results. Returns the "
                        "executor's summary; any plots are shown to the user automatically."),
        "parameters": {"type": "object", "properties": {
            "brief": {"type": "string",
                      "description": "Self-contained task instruction for the executor."},
            "tool_names": {"type": "array", "items": {"type": "string"},
                           "description": "Tool names the executor may use."}},
            "required": ["brief", "tool_names"]}}},
]

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a seismic modeling assistant.
You cannot run seismic tools yourself. Instead you decompose the user's request into task
briefs, discover the right tools for each (discover_tools), delegate each task to an
executor subagent (run_task), then compose ONE final answer.

Rules:
- Call discover_tools before run_task; pick tool names only from its results.
- Make each brief self-contained: repeat every parameter value the user gave.
- Executors share session context: a later task can rely on an earlier task's stored
  result (e.g. the earth model built by the previous run_task), so do NOT repeat work.
- A user message beginning "[image attached: ...]" means a photo was uploaded this turn:
  delegate to the outcrop tools. Never pass image_path, interpretation or model values in
  any brief — they are supplied automatically from session context.
- After an interpret_outcrop task, report the regions and the scale estimate WITH its
  confidence, and ask the user to confirm the height before building the model if
  confidence is low or no scale was found.
- Any plot an executor produces is displayed to the user automatically — never mention
  image file paths.
- In your final answer, state the key quantitative results (tuning thickness, AVO class,
  intercept/gradient, etc.).
- Place your final user-facing answer in <reply></reply> XML tags.

{context_line}"""


class SeismicOrchestrator:
    def __init__(self, llm_client=None, tool_manager=None,
                 knowledge_base=None, tool_index=None):
        self.llm_client = llm_client or LLMClient()
        self.tool_manager = tool_manager or ToolManager()
        self.knowledge_base = knowledge_base or KnowledgeBase(llm_client=self.llm_client)
        self.tool_index = tool_index or ToolIndex()
        self.context_manager = ContextManager()  # per-session, never shared
        self.session_id = uuid.uuid4().hex
        self.context_manager.trace.session_id = self.session_id
        self._knowledge_router = KnowledgeRouter(self.llm_client, self.knowledge_base, self.context_manager)

    def new_session(self) -> "SeismicOrchestrator":
        return SeismicOrchestrator(llm_client=self.llm_client,
                                   tool_manager=self.tool_manager,
                                   knowledge_base=self.knowledge_base,
                                   tool_index=self.tool_index)

    def attach_image(self, path: str) -> None:
        self.context_manager.set_context("last_image", path)

    def _system_prompt(self) -> str:
        keys = sorted(self.context_manager.conversation_context.keys())
        line = f"Session context currently holds: {', '.join(keys)}." if keys \
            else "Session context is empty (fresh conversation)."
        return ORCHESTRATOR_SYSTEM_PROMPT.format(context_line=line)

    def process_single_input(self, user_input: str) -> Dict[str, Any]:
        trace = self.context_manager.trace
        trace.begin_turn(user_input)
        try:
            if self._knowledge_router.is_knowledge_question(user_input):
                reply = self._knowledge_router.handle_knowledge_question(user_input)
                images: List[str] = []
            else:
                result = self._run_meta_loop(user_input)
                reply, images = result["reply"], result["images"]
            if isinstance(reply, bool):
                reply = str(reply)
            elif reply is None:
                reply = "I didn't get a response. Please try again."
            if not reply and images:
                reply = "Here are the results."
            return {"reply": reply, "images": images, "trace": trace.end_turn()}
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            trace.emit("turn_error", error=str(e))
            return {"reply": f"I encountered an error: {str(e)}", "images": [],
                    "trace": trace.end_turn()}

    def _run_meta_loop(self, user_input: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": user_input}]
        images: List[str] = []
        for _ in range(MAX_ORCH_ROUNDS):
            response = self.llm_client.get_completion(
                system_prompt=self._system_prompt(), user_prompt="",
                tools=META_TOOLS, messages=messages)
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])
            emit_event(self.context_manager, "llm",
                       model=response.get("model"),
                       latency_ms=response.get("latency_ms"),
                       tool_call=bool(response.get("tool_calls")),
                       **usage_dict(response.get("usage")))
            if not response.get("tool_calls"):
                messages.append({"role": "assistant", "content": response["content"]})
                reply = extract_reply(response["content"]) or response["content"]
                return {"reply": reply, "images": images}
            tool_call = response["tool_calls"][0]
            messages.append({"role": "assistant", "content": response["content"],
                             "tool_calls": [tool_call]})
            content = self._dispatch_meta(tool_call, images)
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                             "content": content})
        logger.warning(f"Orchestrator round budget ({MAX_ORCH_ROUNDS}) exhausted; "
                       f"forcing tool-free completion")
        emit_event(self.context_manager, "budget_exhausted", rounds=MAX_ORCH_ROUNDS)
        final_response = self.llm_client.get_completion(
            system_prompt=self._system_prompt(), user_prompt="",
            tools=None, messages=messages)
        if final_response.get("usage"):
            self.context_manager.update_token_usage(final_response["usage"])
        emit_event(self.context_manager, "llm",
                   model=final_response.get("model"),
                   latency_ms=final_response.get("latency_ms"),
                   tool_call=bool(final_response.get("tool_calls")),
                   **usage_dict(final_response.get("usage")))
        reply = extract_reply(final_response["content"]) or final_response["content"]
        return {"reply": reply, "images": images}

    def _dispatch_meta(self, tool_call, images: List[str]) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments) \
                if isinstance(tool_call.function.arguments, str) \
                else dict(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return f"Invalid arguments: {e}"
        if not isinstance(args, dict):
            return "Invalid arguments: expected a JSON object."
        if name == "discover_tools":
            return self._discover(args.get("task_description", ""))
        if name == "run_task":
            return self._run_task(args.get("brief", ""),
                                  args.get("tool_names") or [], images)
        return f"Unknown meta-tool: {name}. Use discover_tools or run_task."

    def _discover(self, task_description: str) -> str:
        cards = self.tool_index.search(task_description)
        hits = [[c.name, round(c.score, 4)] for c in cards]
        logger.info(f"discover_tools({task_description!r}) -> {hits}")
        emit_event(self.context_manager, "discover",
                   query=task_description[:200], hits=hits)
        if not cards:
            return "No tools matched; rephrase the task or answer directly."
        return "Matching tools:\n" + "\n".join(f"- {c.card}" for c in cards)

    def _run_task(self, brief: str, tool_names: List[str], images: List[str]) -> str:
        if not isinstance(tool_names, list) or not tool_names:
            return "tool_names is empty — call discover_tools first."
        unknown = [n for n in tool_names if n not in REGISTRY_BY_NAME]
        if unknown:
            return (f"Unknown tool name(s): {', '.join(unknown)}. "
                    f"Use names exactly as returned by discover_tools.")
        executor = ExecutorAgent(self.llm_client, self.tool_manager, self.context_manager)
        result = executor.run(brief, tool_names)
        emit_event(self.context_manager, "run_task", brief=brief[:200],
                   tool_names=list(tool_names), tools_used=result.tools_used,
                   error=result.error, n_images=len(result.images))
        for p in result.images:
            if p not in images:
                images.append(p)
        payload = {"summary": result.summary, "tools_used": result.tools_used}
        if result.error:
            payload["error"] = result.error
        if result.images:
            payload["plots"] = f"{len(result.images)} plot(s) shown to the user"
        return json.dumps(payload)
