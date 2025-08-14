import logging
import re
import json
import numpy as np
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient
from .tool_manager import ToolManager
from .context_manager import ContextManager
from knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class SeismicChatBotToolUse:
    """
    Seismic ChatBot using the tool use pattern from the notebook.
    This implementation follows the same flow as the example:
    1. System prompt with tool definitions
    2. Tool use flow with proper message handling
    3. Tool execution and result processing
    """
    
    def __init__(self):
        """Initialize the seismic chatbot with all required components."""
        self.llm_client = LLMClient()
        self.tool_manager = ToolManager()
        self.context_manager = ContextManager()
        self.knowledge_base = KnowledgeBase()
        
        # Get tool schemas for the LLM
        self.tools = self.tool_manager.get_tool_schemas()
        
        # System prompt following the notebook pattern
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt following the notebook pattern.
        
        Returns:
            str: The system prompt
        """
        return """
You are a seismic modeling assistant chatbot. Your job is to help users with seismic analysis, 
wavelet generation, wedge modeling, and AVO calculations. You have access to a set of tools, 
but only use them when needed.

Available tools:
- make_ricker: Creates a Ricker wavelet with specified frequency
- plot_ricker: Plots a Ricker wavelet with time and frequency analysis
- wedge_model: Creates a wedge model for seismic analysis
- plot_wedge_model: Plots wedge model results
- zoeppritz_reflectivity: Calculates reflectivity using Zoeppritz equations
- shuey_reflectivity: Calculates reflectivity using Shuey's approximation
- plot_avo_reflectivity: Plots AVO reflectivity curves

Guidelines:
1. Be helpful and concise in your responses
2. Only use tools when you have all required parameters
3. If you don't have enough information to use a tool correctly, ask follow-up questions
4. For seismic questions, provide educational explanations
5. When using tools, explain what you're doing and interpret the results

In each conversational turn, you will:
1. Think about the user's request
2. Use tools if needed and you have the required parameters
3. Provide a clear, helpful response

Place all user-facing conversational responses in <reply></reply> XML tags to make them easy to parse.
"""

    def _parse_tool_input(self, tool_input: str) -> Dict[str, Any]:
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

    def chat(self, user_input: str = None) -> str:
        """
        Main chat function following the notebook pattern.
        
        Args:
            user_input: Initial user input (if None, will prompt for input)
            
        Returns:
            str: The chatbot's response
        """
        messages = []
        
        if user_input is None:
            user_input = input("\nUser: ")
        
        messages.append({"role": "user", "content": user_input})
        
        while True:
            if user_input.lower() == "quit":
                break
                
            # If the last message is from the assistant, get another input from the user
            if messages[-1].get("role") == "assistant":
                user_input = input("\nUser: ")
                messages.append({"role": "user", "content": user_input})
                if user_input.lower() == "quit":
                    break

            # Send request to LLM
            response = self.llm_client.get_completion(
                system_prompt=self.system_prompt,
                user_prompt="",
                tools=self.tools,
                messages=messages
            )
            
            # Update token usage statistics
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])
            
            # If LLM stops because it wants to use a tool
            if response["stop_reason"] == "tool_calls" and response["tool_calls"]:
                # Append the assistant message with tool_calls
                messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": response["tool_calls"]
                })
                # Handle tool calls (assuming single tool call for simplicity)
                tool_call = response["tool_calls"][0]
                tool_name = tool_call.function.name
                tool_input_str = tool_call.function.arguments
                print(f"=====Using the {tool_name} tool=====")
                try:
                    # Parse tool input
                    tool_input = self._parse_tool_input(tool_input_str)
                    # Supplement plot_ricker parameters from context if missing
                    if tool_name == "plot_ricker":
                        last_wavelet = self.context_manager.get_context("last_ricker_wavelet")
                        if last_wavelet:
                            if "wavelet" not in tool_input:
                                tool_input["wavelet"] = last_wavelet.get("wavelet")
                            if "time_array" not in tool_input:
                                tool_input["time_array"] = last_wavelet.get("time_array")
                    # Execute the tool
                    tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                    # Add tool result to messages (role: tool)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                    # Update context with tool result
                    self._update_context(tool_name, tool_input, tool_result)
                    # Now, get the next assistant response
                    final_response = self.llm_client.get_completion(
                        system_prompt=self.system_prompt,
                        user_prompt="",
                        tools=self.tools,
                        messages=messages
                    )
                    
                    # Update token usage statistics
                    if final_response.get("usage"):
                        self.context_manager.update_token_usage(final_response["usage"])
                    messages.append({"role": "assistant", "content": final_response["content"]})
                    model_reply = self._extract_reply(final_response["content"])
                    if model_reply:
                        print(f"\nSeismic Assistant: {model_reply}")
                    else:
                        print(f"\nSeismic Assistant: {final_response['content']}")
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    print(f"Error executing tool: {str(e)}")
            else:
                # If LLM does NOT want to use a tool, extract and print the response
                messages.append({"role": "assistant", "content": response["content"]})
                model_reply = self._extract_reply(response["content"])
                if model_reply:
                    print(f"\nSeismic Assistant: {model_reply}")
                else:
                    print(f"\nSeismic Assistant: {response['content']}")

    def process_single_input(self, user_input: str) -> str:
        """
        Process a single user input and return the response.
        This is useful for API interfaces.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: The chatbot's response
        """
        messages = [{"role": "user", "content": user_input}]
        
        # Send request to LLM
        response = self.llm_client.get_completion(
            system_prompt=self.system_prompt,
            user_prompt="",
            tools=self.tools,
            messages=messages
        )
        
        # Update token usage statistics
        if response.get("usage"):
            self.context_manager.update_token_usage(response["usage"])
        # If LLM wants to use a tool
        if response.get("tool_calls"):
            # Append the assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response["tool_calls"]
            })
            tool_call = response["tool_calls"][0]
            tool_name = tool_call.function.name
            tool_input_str = tool_call.function.arguments
            try:
                # Parse tool input
                tool_input = self._parse_tool_input(tool_input_str)
                # Supplement plot_ricker parameters from context if missing
                if tool_name == "plot_ricker":
                    last_wavelet = self.context_manager.get_context("last_ricker_wavelet")
                    if last_wavelet:
                        if "wavelet" not in tool_input:
                            tool_input["wavelet"] = last_wavelet.get("wavelet")
                        if "time_array" not in tool_input:
                            tool_input["time_array"] = last_wavelet.get("time_array")
                # Execute the tool
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                # Add tool result to messages (role: tool)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
                # Update context
                self._update_context(tool_name, tool_input, tool_result)
                # Special handling for plot_ricker image output
                if tool_name == "plot_ricker" and isinstance(tool_result, str) and tool_result.endswith(".png"):
                    return {"image_path": tool_result}
                # --- Automatic chaining: if make_ricker, immediately call plot_ricker ---
                if tool_name == "make_ricker":
                    # Get the last generated wavelet from context
                    last_wavelet = self.context_manager.get_context("last_ricker_wavelet")
                    if last_wavelet:
                        plot_input = {
                            "wavelet": last_wavelet["wavelet"],
                            "time_array": last_wavelet["time_array"]
                        }
                        plot_result = self.tool_manager.process_tool_call("plot_ricker", plot_input)
                        if isinstance(plot_result, str) and plot_result.endswith(".png"):
                            return {"image_path": plot_result}
                # --- Automatic chaining: if wedge_model, immediately call plot_wedge_model ---
                elif tool_name == "wedge_model":
                    # Get the last generated wedge model from context
                    last_wedge = self.context_manager.get_context("last_wedge_model")
                    if last_wedge and "synthetic" in last_wedge and "parameters" in last_wedge:
                        plot_input = {
                            "synthetic_data": last_wedge["synthetic"],
                            "parameters": last_wedge["parameters"]
                        }
                        plot_result = self.tool_manager.process_tool_call("plot_wedge_model", plot_input)
                        if isinstance(plot_result, str) and plot_result.endswith(".png"):
                            return {"image_path": plot_result}
                # --- Automatic chaining: if zoeppritz_reflectivity or shuey_reflectivity, immediately call plot_avo_reflectivity ---
                elif tool_name in ["zoeppritz_reflectivity", "shuey_reflectivity"]:
                    # Get the reflection coefficients and angles from the tool input
                    if isinstance(tool_result, np.ndarray) and "angles" in tool_input:
                        plot_input = {
                            "angles": tool_input["angles"],
                            "rc": tool_result
                        }
                        plot_result = self.tool_manager.process_tool_call("plot_avo_reflectivity", plot_input)
                        if isinstance(plot_result, str) and plot_result.endswith(".png"):
                            return {"image_path": plot_result}
                # --- End automatic chaining ---
                final_response = self.llm_client.get_completion(
                    system_prompt=self.system_prompt,
                    user_prompt="",
                    tools=self.tools,
                    messages=messages
                )
                
                # Update token usage statistics
                if final_response.get("usage"):
                    self.context_manager.update_token_usage(final_response["usage"])
                result = self._extract_reply(final_response["content"]) or final_response["content"]
                return result
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return f"Error executing tool: {str(e)}"
        else:
            # Return the direct response
            messages.append({"role": "assistant", "content": response["content"]})
            result = self._extract_reply(response["content"]) or response["content"]
            return result

    def _extract_reply(self, text: str) -> Optional[str]:
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

    def _update_context(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any):
        """
        Update conversation context with tool execution results.
        
        Args:
            tool_name: Name of the tool executed
            tool_input: Input parameters used
            tool_result: Result from tool execution
        """
        try:
            if tool_name == "make_ricker":
                # Store frequency for future use
                if "frequency" in tool_input:
                    self.context_manager.set_context("last_frequency", tool_input["frequency"])
                
                # Store wavelet data
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
                    
            elif tool_name in ["zoeppritz_reflectivity", "shuey_reflectivity"]:
                # Store AVO reflectivity data for reference
                if isinstance(tool_result, np.ndarray) and "angles" in tool_input:
                    self.context_manager.set_context("last_avo_reflectivity", {
                        "angles": tool_input["angles"],
                        "rc": tool_result,
                        "method": tool_name,
                        "parameters": tool_input
                    })
                
        except Exception as e:
            logger.error(f"Error updating context: {e}")

    def get_available_tools(self) -> List[Dict]:
        """
        Get list of available tools.
        
        Returns:
            List[Dict]: List of tool schemas
        """
        return self.tools