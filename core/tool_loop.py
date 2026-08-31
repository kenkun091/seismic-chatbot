"""Shared bounded tool-use loop.

One implementation of parse → inject-context → execute → compact → auto-plot →
harvest, used by both the classic SeismicChatBotToolUse and the agentic-mode
ExecutorAgent. Extracted verbatim from chatbot_tool_use.py — behavior changes
here change BOTH bots; keep it that way.
"""
import logging
import re
import json
import time
import warnings
import numpy as np
from typing import Dict, Any, List, Optional
from core.tool_registry import AUTO_PLOT
from core.turn_trace import emit_event, usage_dict
from workflows.engine import WORKFLOW_NAMES

logger = logging.getLogger(__name__)

# Numeric sequences longer than this are summarized before being sent back to
# the LLM as tool-message content (narration needs the stats, not 61 floats).
_MAX_ARRAY_PREVIEW = 12


def extract_reply(text: str) -> Optional[str]:
    """
    Extract reply from XML tags following the notebook pattern.

    Args:
        text: Text containing XML tags

    Returns:
        Optional[str]: Extracted reply or None
    """
    pattern = r'<reply>(.*?)</reply>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


class ToolLoopRunner:
    """The bounded agentic tool loop, extracted so it can be shared."""

    # Tools whose heavy inputs live in per-session context rather than in the
    # LLM's arguments: (tool name, parameter name, context key).
    _CONTEXT_INPUTS = (
        ("interpret_outcrop", "image_path", "last_image"),
        ("outcrop_to_seismic", "image_path", "last_image"),
        ("outcrop_to_model", "interpretation", "last_outcrop"),
        ("synthetic_section", "model", "last_earth_model"),
    )

    def __init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        self.max_tool_rounds = max_tool_rounds

    def parse_tool_input(self, tool_input: str) -> Dict[str, Any]:
        """
        Parse tool input from JSON string to dictionary.

        Args:
            tool_input: JSON string or dictionary

        Returns:
            Dict[str, Any]: Parsed tool input
        """
        if isinstance(tool_input, dict):
            return tool_input
        elif isinstance(tool_input, str):
            try:
                return json.loads(tool_input)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool input JSON: {e}")
                raise ValueError(f"Invalid tool input format: {e}")
        else:
            raise ValueError(f"Unexpected tool input type: {type(tool_input)}")

    def compact_tool_result(self, tool_result: Any) -> str:
        """Compact a tool result for the LLM's role:"tool" message.

        Large numeric arrays become summary strings and image paths are masked
        — plots are displayed to the user directly, so the model should narrate
        the numbers, not echo file paths.
        """
        compacted = self.compact_value(tool_result)
        try:
            return json.dumps(compacted, default=str)
        except (TypeError, ValueError):
            return str(compacted)

    def compact_value(self, value: Any) -> Any:
        """Recursively compact one value (see compact_tool_result)."""
        if isinstance(value, np.ndarray):
            if value.size > _MAX_ARRAY_PREVIEW:
                return (f"<array shape {value.shape}, "
                        f"min={value.min():.6g}, max={value.max():.6g}>")
            value = value.tolist()
        if isinstance(value, dict):
            return {
                k: ("<plot generated and shown to the user>"
                    if k == "image_path" and isinstance(v, str)
                    else self.compact_value(v))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            seq = list(value)
            if (len(seq) > _MAX_ARRAY_PREVIEW
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            for x in seq)):
                arr = [float(x) for x in seq]
                return (f"<{len(arr)} values, min={min(arr):.6g}, max={max(arr):.6g}, "
                        f"first={arr[0]:.6g}, last={arr[-1]:.6g}>")
            return [self.compact_value(v) for v in seq]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def inject_context_inputs(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Fill omitted context-resident parameters from the session context.

        For `last_image`, the session's attached photo always wins: the LLM
        cannot know the staged sandbox filename, so a differing LLM-supplied
        `image_path` is logged and overridden rather than trusted. The other
        context-resident params (interpretation, model) keep the weaker
        "fill only when the LLM left it absent" behaviour.
        """
        filled = dict(tool_input)
        for name, param, key in self._CONTEXT_INPUTS:
            if name != tool_name:
                continue
            value = self.context_manager.get_context(key)
            if key == "last_image":
                if value is not None:
                    if filled.get(param) is not None and filled[param] != value:
                        logger.warning(
                            f"{tool_name}: ignoring LLM-supplied {param}={filled[param]!r}; "
                            f"using the session's uploaded image {value!r} instead"
                        )
                    filled[param] = value
                elif filled.get(param) is None:
                    filled.pop(param, None)
            elif filled.get(param) is None:
                if value is not None:
                    filled[param] = value
                else:
                    filled.pop(param, None)
        return filled

    def harvest_images(self, tool_result: Any, collected: List[str]) -> None:
        """Collect .png paths from a tool result into `collected`.

        Handles the two shapes tools produce: a plain path string (plot tools)
        or a dict carrying an "image_path" key (workflow recipes, auto-chain
        results). Deduped, order-preserving. Only a TOP-LEVEL "image_path" is
        collected — nested dicts are not recursed (every current recipe returns
        a single top-level composite plot; revisit if one ever nests plots).
        A top-level "extra_image_paths" list (outcrop_to_seismic's overlay) is
        collected too. A dict's "image_path" is skipped when it equals the
        session's last_image: interpret_outcrop's result echoes the user's
        uploaded photo under that same key, and that is not a generated plot.
        """
        last_image = self.context_manager.get_context("last_image")
        paths = []
        if isinstance(tool_result, str) and tool_result.endswith(".png"):
            paths.append(tool_result)
        elif isinstance(tool_result, dict):
            p = tool_result.get("image_path")
            if isinstance(p, str) and p.endswith(".png") and p != last_image:
                paths.append(p)
            for extra in tool_result.get("extra_image_paths") or []:
                if isinstance(extra, str) and extra.endswith(".png"):
                    paths.append(extra)
        for path in paths:
            if path not in collected:
                collected.append(path)

    def handle_automatic_chaining(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any) -> Optional[Dict[str, Any]]:
        """
        Handle automatic chaining of related tools.

        Args:
            tool_name: Name of the executed tool
            tool_input: Input parameters for the tool
            tool_result: Result from the tool

        Returns:
            Optional dict with image path if chaining occurred
        """
        plot_tool = AUTO_PLOT.get(tool_name)
        if plot_tool is None:
            return None
        try:
            if tool_name in ("make_ricker", "make_ormsby"):
                last = self.context_manager.get_context("last_ricker_wavelet")
                if not last:
                    return None
                plot_input = {"wavelet": last["wavelet"], "time_array": last["time_array"]}
            elif tool_name == "wedge_model":
                last = self.context_manager.get_context("last_wedge_model")
                if not (last and "synthetic" in last and "parameters" in last):
                    return None
                plot_input = {"synthetic_data": last["synthetic"], "parameters": last["parameters"]}
            elif tool_name == "wedge_avo_gather":
                last = self.context_manager.get_context("last_wedge_gather")
                if not (last and "gather" in last and "parameters" in last):
                    return None
                plot_input = {"gather": last["gather"], "parameters": last["parameters"]}
            elif tool_name == "synthetic_seismogram":
                last = self.context_manager.get_context("last_synthetic")
                if not (last and "trace" in last and "parameters" in last):
                    return None
                plot_input = {"trace": last["trace"], "parameters": last["parameters"]}
            elif tool_name in ("zoeppritz_reflectivity", "shuey_reflectivity"):
                if not (isinstance(tool_result, np.ndarray) and "angles" in tool_input):
                    return None
                plot_input = {"angles": tool_input["angles"], "rc": tool_result}
            elif tool_name == "avo_attributes":
                if not (isinstance(tool_result, dict) and "intercept" in tool_result):
                    return None
                plot_input = {
                    "intercept": tool_result["intercept"],
                    "gradient": tool_result["gradient"],
                    "avo_class": tool_result.get("avo_class"),
                }
            elif tool_name == "extended_elastic_impedance":
                if not (isinstance(tool_result, np.ndarray) and "chi" in tool_input):
                    return None
                plot_input = {"chi": tool_input["chi"], "eei": tool_result}
            elif tool_name == "calculate_rock_properties":
                last = self.context_manager.get_context("last_rock_properties")
                if not last:
                    return None
                plot_input = {
                    "phit": last["phit"],
                    "vclay": last["vclay"],
                    "vp": last["vp"],
                    "vs": last["vs"],
                    "rhob": last["rhob"],
                    "vp_vs_ratio": last["vp_vs_ratio"],
                    "ai": last["acoustic_impedance"],
                    "si": last["shear_impedance"],
                    "fluid_type": last.get("fluid_type", "water"),
                }
            elif tool_name == "interpret_outcrop":
                last = self.context_manager.get_context("last_outcrop")
                if not last:
                    return None
                plot_input = {"interpretation": last}
            elif tool_name == "synthetic_section":
                last = self.context_manager.get_context("last_section")
                if not (last and "section" in last and "parameters" in last):
                    return None
                plot_input = {
                    "section": last["section"],
                    "parameters": last["parameters"],
                    "axis": last.get("axis"),
                    "model": self.context_manager.get_context("last_earth_model"),
                    "display": last["parameters"].get("display", "overlay"),
                }
            else:
                return None

            plot_result = self.tool_manager.process_tool_call(plot_tool, plot_input)
            if isinstance(plot_result, str) and plot_result.endswith(".png"):
                return {"image_path": plot_result}
            return None
        except Exception as e:
            logger.error(f"Error in automatic chaining: {e}", exc_info=True)
            return None

    def update_context(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any):
        """
        Update conversation context with tool execution results.

        Args:
            tool_name: Name of the tool executed
            tool_input: Input parameters used
            tool_result: Result from tool execution
        """
        try:
            if tool_name in ("make_ricker", "make_ormsby"):
                # Store frequency for future use (only for make_ricker which has a single frequency)
                if tool_name == "make_ricker" and "frequency" in tool_input:
                    self.context_manager.set_context("last_frequency", tool_input["frequency"])

                # Store wavelet data for both make_ricker and make_ormsby (same tuple shape)
                if isinstance(tool_result, tuple) and len(tool_result) == 2:
                    time_array, wavelet = tool_result
                    self.context_manager.set_context("last_ricker_wavelet", {
                        "time_array": time_array,
                        "wavelet": wavelet,
                        "parameters": tool_input
                    })

            elif tool_name == "wedge_model":
                # Store wedge model data for automatic plotting
                if isinstance(tool_result, tuple) and len(tool_result) == 4:
                    time_array, model, synthetic, parameters = tool_result
                    self.context_manager.set_context("last_wedge_model", {
                        "time_array": time_array,
                        "model": model,
                        "synthetic": synthetic,
                        "parameters": parameters,
                        "input_params": tool_input
                    })

            elif tool_name == "wedge_avo_gather":
                # Store gather data for automatic plotting (3-tuple return)
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    time_array, gather, parameters = tool_result
                    self.context_manager.set_context("last_wedge_gather", {
                        "time_array": time_array,
                        "gather": gather,
                        "parameters": parameters,
                        "input_params": tool_input
                    })

            elif tool_name == "synthetic_seismogram":
                # Store synthetic trace for automatic plotting (3-tuple return)
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    time_array, trace, parameters = tool_result
                    self.context_manager.set_context("last_synthetic", {
                        "time_array": time_array,
                        "trace": trace,
                        "parameters": parameters,
                        "input_params": tool_input
                    })

            elif tool_name in ["zoeppritz_reflectivity", "shuey_reflectivity"]:
                # Store AVO reflectivity data for reference
                if isinstance(tool_result, np.ndarray) and "angles" in tool_input:
                    self.context_manager.set_context("last_avo_reflectivity", {
                        "angles": tool_input["angles"],
                        "rc": tool_result,
                        "method": tool_name,
                        "parameters": tool_input
                    })

            elif tool_name == "avo_attributes":
                if isinstance(tool_result, dict) and "intercept" in tool_result:
                    self.context_manager.set_context("last_avo_attributes", tool_result)

            elif tool_name == "extended_elastic_impedance":
                if isinstance(tool_result, np.ndarray) and "chi" in tool_input:
                    self.context_manager.set_context("last_eei", {
                        "chi": tool_input["chi"],
                        "eei": tool_result,
                        "parameters": tool_input,
                    })

            elif tool_name == "calculate_rock_properties":
                # Store rock properties data for reference
                if isinstance(tool_result, tuple) and len(tool_result) == 6:
                    vp, vs, rhob, vp_vs_ratio, ai, si = tool_result
                    self.context_manager.set_context("last_rock_properties", {
                        "phit": tool_input["phit"],
                        "vclay": tool_input["vclay"],
                        "vp": vp,
                        "vs": vs,
                        "rhob": rhob,
                        "vp_vs_ratio": vp_vs_ratio,
                        "acoustic_impedance": ai,
                        "shear_impedance": si,
                        "fluid_type": tool_input.get("fluid_type", "water"),
                        "parameters": tool_input
                    })

            elif tool_name == "interpret_outcrop":
                if isinstance(tool_result, dict) and "regions" in tool_result:
                    self.context_manager.set_context("last_outcrop", tool_result)

            elif tool_name == "outcrop_to_model":
                if isinstance(tool_result, dict) and "facies" in tool_result:
                    self.context_manager.set_context("last_earth_model", tool_result)

            elif tool_name == "synthetic_section":
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    axis, section, parameters = tool_result
                    self.context_manager.set_context("last_section", {
                        "axis": axis,
                        "section": section,
                        "parameters": parameters,
                        "input_params": tool_input
                    })

            elif tool_name == "outcrop_to_seismic":
                if isinstance(tool_result, dict):
                    self.context_manager.set_context("last_workflow_result", tool_result)
                    if tool_result.get("interpretation") is not None:
                        self.context_manager.set_context("last_outcrop", tool_result["interpretation"])
                    if tool_result.get("model") is not None:
                        self.context_manager.set_context("last_earth_model", tool_result["model"])
                    sec = tool_result.get("section")
                    if isinstance(sec, dict):
                        self.context_manager.set_context("last_section", {
                            "axis": sec.get("axis"),
                            "section": sec.get("section"),
                            "parameters": sec.get("parameters"),
                            "input_params": tool_input
                        })

            elif tool_name in WORKFLOW_NAMES:
                if isinstance(tool_result, dict):
                    self.context_manager.set_context("last_workflow_result", tool_result)

        except Exception as e:
            logger.error(f"Error updating context: {e}")

    def _emit_llm(self, response: Dict[str, Any]) -> None:
        emit_event(self.context_manager, "llm",
                   model=response.get("model"),
                   latency_ms=response.get("latency_ms"),
                   tool_call=bool(response.get("tool_calls")),
                   **usage_dict(response.get("usage")))

    def run(self, system_prompt: str, messages: List[dict], tools: list) -> Dict[str, Any]:
        """The bounded loop from _handle_tool_request, generalized.

        Differences from the original method: system prompt and tools are
        parameters; successfully executed tool names are recorded in
        'tools_used'; reply extraction uses module-level extract_reply.
        """
        collected_images: List[str] = []
        tools_used: List[str] = []

        # Agentic tool loop: the model may chain several tool calls before
        # giving a final answer. Plots are harvested into collected_images and
        # a compacted tool result goes back to the model so it can narrate the
        # numbers (bounded to avoid runaways).
        for _ in range(self.max_tool_rounds):
            response = self.llm_client.get_completion(
                system_prompt=system_prompt,
                user_prompt="",
                tools=tools,
                messages=messages
            )
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])
            self._emit_llm(response)

            if not response.get("tool_calls"):
                # No tool requested: this is the final answer.
                messages.append({"role": "assistant", "content": response["content"]})
                reply = extract_reply(response["content"]) or response["content"]
                if isinstance(reply, bool):
                    reply = str(reply)
                return {"reply": reply, "images": collected_images, "tools_used": tools_used}

            # Execute the (first) requested tool. Append only the tool_call we
            # respond to so every assistant tool_call has a matching tool result.
            tool_call = response["tool_calls"][0]
            if len(response["tool_calls"]) > 1:
                dropped = [tc.function.name for tc in response["tool_calls"][1:]]
                logger.warning(
                    f"Executing only the first of {len(response['tool_calls'])} "
                    f"requested tool calls; dropped: {dropped}")
                emit_event(self.context_manager, "parallel_calls_dropped", dropped=dropped)
            tool_name = tool_call.function.name
            tool_input_str = tool_call.function.arguments
            messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": [tool_call]
            })

            try:
                raw_input = self.parse_tool_input(tool_input_str)
                tool_input = self.inject_context_inputs(tool_name, raw_input)
                injected = sorted(k for k in tool_input if k not in raw_input)
                overridden = sorted(
                    k for k in raw_input
                    if k in tool_input
                    and isinstance(raw_input.get(k), str)
                    and isinstance(tool_input.get(k), str)
                    and raw_input[k] != tool_input[k])
                spec = getattr(self.tool_manager, "specs", {}).get(tool_name)
                defaults_filled = sorted(
                    k for k in spec.defaults if k not in tool_input) if spec else []
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                for w in caught:
                    message = str(w.message)[:300]
                    logger.warning(f"{tool_name}: {w.category.__name__}: {message}")
                    emit_event(self.context_manager, "physics_warning",
                               tool=tool_name, category=w.category.__name__,
                               message=message)
                emit_event(self.context_manager, "tool_call", tool=tool_name, ok=True,
                           ms=round((time.perf_counter() - started) * 1000, 1),
                           injected=injected, overridden=overridden,
                           defaults_filled=defaults_filled)
                tools_used.append(tool_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self.compact_tool_result(tool_result)
                })
                self.update_context(tool_name, tool_input, tool_result)
                self.harvest_images(tool_result, collected_images)

                # Auto-chaining still runs the partner plot tool; its plot now
                # joins the harvest instead of ending the turn.
                chained_result = self.handle_automatic_chaining(tool_name, tool_input, tool_result)
                if chained_result:
                    self.harvest_images(chained_result, collected_images)
                    emit_event(self.context_manager, "auto_plot", compute=tool_name,
                               plot=AUTO_PLOT.get(tool_name), fired=True)
                elif AUTO_PLOT.get(tool_name):
                    logger.warning(
                        f"auto-plot {AUTO_PLOT[tool_name]} did not run after "
                        f"{tool_name} (missing context or plot error)")
                    emit_event(self.context_manager, "auto_plot", compute=tool_name,
                               plot=AUTO_PLOT[tool_name], fired=False)

                # Loop so the model can narrate the result or chain another tool.
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
                emit_event(self.context_manager, "tool_call", tool=tool_name,
                           ok=False, error=str(e))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"Tool execution failed: {e}. Do not retry with the same "
                        f"arguments; summarize what you have or ask the user for "
                        f"clarification."
                    ),
                })
                continue

        logger.warning(
            f"Tool-round budget ({self.max_tool_rounds}) exhausted; forcing "
            f"tool-free completion")
        emit_event(self.context_manager, "budget_exhausted", rounds=self.max_tool_rounds, scope="tool_loop")

        # Round budget exhausted while still calling tools: force a tool-free
        # completion so the user gets a textual answer instead of nothing.
        final_response = self.llm_client.get_completion(
            system_prompt=system_prompt,
            user_prompt="",
            tools=None,
            messages=messages
        )
        if final_response.get("usage"):
            self.context_manager.update_token_usage(final_response["usage"])
        self._emit_llm(final_response)
        reply = extract_reply(final_response["content"]) or final_response["content"]
        if isinstance(reply, bool):
            reply = str(reply)
        return {"reply": reply, "images": collected_images, "tools_used": tools_used}
