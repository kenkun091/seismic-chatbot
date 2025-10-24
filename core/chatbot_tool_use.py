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
        Process a single user input and return a response.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: The chatbot's response
        """
        try:
            # Check if this is a knowledge question that should use RAG
            if self._is_knowledge_question(user_input):
                logger.info("Using RAG for knowledge question")
                response = self._handle_knowledge_question(user_input)
            else:
                # Otherwise, use the regular tool-based approach
                logger.info("Using tool-based approach")
                response = self._handle_tool_request(user_input)
            
            # Final safety check: ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            elif response is None:
                response = "I didn't get a response. Please try again."
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _is_knowledge_question(self, user_input: str) -> bool:
        """
        Determine if the user input is a knowledge question using LLM-based intent classification.
        
        Args:
            user_input: The user's input text
            
        Returns:
            bool: True if this should use RAG
        """
        try:
            # Use LLM for intent classification
            return self._classify_intent_with_llm(user_input)
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            # Fallback to keyword-based detection
            return self._is_knowledge_question_keywords(user_input)
    
    def _classify_intent_with_llm(self, user_input: str) -> bool:
        """
        Use LLM to classify whether the input is a knowledge question.
        
        Args:
            user_input: The user's input text
            
        Returns:
            bool: True if this is a knowledge question that should use RAG
        """
        system_prompt = """You are an expert at classifying user intents in a seismic modeling chatbot context.

Your task is to determine if a user's input is a KNOWLEDGE QUESTION that should be answered using the knowledge base (RAG), or if it's a TOOL REQUEST that should use specific seismic modeling tools.

KNOWLEDGE QUESTIONS include:
- Questions asking for explanations, definitions, or descriptions
- Questions about concepts, principles, or theory
- Questions about relationships, effects, or trade-offs
- Questions starting with "what", "how", "why", "explain", "describe", "tell me about"
- Questions about seismic properties, resolution, frequency effects, etc.
- Questions seeking educational information or understanding

TOOL REQUESTS include:
- Requests to create, generate, or make something (e.g., "create a Ricker wavelet")
- Requests to plot, visualize, or display something
- Requests to calculate or compute specific values
- Requests to model or simulate something
- Requests with specific parameters or values

Examples:
- "How does frequency affect seismic resolution?" → KNOWLEDGE QUESTION
- "What is a Ricker wavelet?" → KNOWLEDGE QUESTION  
- "Create a 30 Hz Ricker wavelet" → TOOL REQUEST
- "Plot the wedge model" → TOOL REQUEST
- "What are the trade-offs of higher frequency?" → KNOWLEDGE QUESTION
- "Make a wedge model with 100m thickness" → TOOL REQUEST

Respond with ONLY "KNOWLEDGE" or "TOOL" - no other text."""

        try:
            response = self.llm_client.get_simple_completion(system_prompt, user_input)
            response = response.strip().upper()
            
            # Log the classification for debugging
            logger.debug(f"LLM classified '{user_input[:50]}...' as: {response}")
            
            return response == "KNOWLEDGE"
            
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
            raise e
    
    def classify_intent_detailed(self, user_input: str) -> Dict[str, Any]:
        """
        Use LLM to classify user intent with detailed information.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict with intent classification and confidence
        """
        system_prompt = """You are an expert at classifying user intents in a seismic modeling chatbot context.

Classify the user's intent and provide detailed information about it.

INTENT TYPES:
1. KNOWLEDGE_QUESTION - Questions seeking explanations, definitions, or educational information
2. TOOL_REQUEST - Requests to create, plot, calculate, or model something
3. MIXED - Both knowledge and tool components
4. UNCLEAR - Ambiguous or unclear intent

For each intent, also determine:
- CONFIDENCE: How confident you are (0.0 to 1.0)
- REASONING: Brief explanation of your classification
- SUGGESTED_ACTION: What the chatbot should do

Respond in JSON format:
{
    "intent": "KNOWLEDGE_QUESTION|TOOL_REQUEST|MIXED|UNCLEAR",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "suggested_action": "Use RAG|Use Tools|Ask for clarification|Both"
}"""

        try:
            response = self.llm_client.get_simple_completion(system_prompt, user_input)
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response)
                logger.debug(f"Detailed classification: {result}")
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {response}")
                return {
                    "intent": "UNCLEAR",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse LLM response",
                    "suggested_action": "Ask for clarification"
                }
                
        except Exception as e:
            logger.error(f"Error in detailed intent classification: {e}")
            return {
                "intent": "UNCLEAR", 
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "suggested_action": "Use fallback classification"
            }
    
    def _is_knowledge_question_keywords(self, user_input: str) -> bool:
        """
        Fallback keyword-based detection for knowledge questions.
        
        Args:
            user_input: The user's input text
            
        Returns:
            bool: True if this should use RAG
        """
        # Keywords that indicate knowledge questions
        knowledge_keywords = [
            'what is', 'what are', 'explain', 'describe', 'how does', 'why does',
            'tell me about', 'what determines', 'what affects', 'what causes',
            'difference between', 'relationship between', 'definition of',
            'characteristics of', 'properties of', 'applications of',
            'how can', 'what happens', 'what is the', 'what are the',
            'can you explain', 'can you describe', 'what do you know',
            'trade-offs', 'advantages', 'disadvantages', 'benefits',
            'limitations', 'constraints', 'factors', 'influence',
            'impact', 'effect', 'role', 'significance', 'importance'
        ]
        
        user_input_lower = user_input.lower()
        
        # Check if input contains knowledge question patterns
        for keyword in knowledge_keywords:
            if keyword in user_input_lower:
                return True
        
        # Check if it's a question (ends with ?)
        if user_input.strip().endswith('?'):
            return True
        
        # Check if it's asking for explanation
        if any(word in user_input_lower for word in ['explain', 'describe', 'tell me']):
            return True
        
        # Check for seismic/geophysical concept questions
        seismic_concepts = [
            'frequency', 'resolution', 'bandwidth', 'wavelength', 'tuning',
            'impedance', 'velocity', 'density', 'attenuation', 'quality factor',
            'reflection', 'refraction', 'wavelet', 'ricker', 'wedge model',
            'seismic', 'geophysical', 'geology', 'petroleum', 'reservoir'
        ]
        
        # If it contains seismic concepts and is asking for information, use RAG
        if any(concept in user_input_lower for concept in seismic_concepts):
            if any(word in user_input_lower for word in ['what', 'how', 'why', 'explain', 'describe', 'tell']):
                return True
        
        return False
    
    def _handle_knowledge_question(self, user_input: str) -> str:
        """
        Handle knowledge questions using RAG.
        
        Args:
            user_input: The user's question
            
        Returns:
            str: Generated response using RAG
        """
        try:
            # Use the knowledge base's RAG system
            rag_response = self.knowledge_base.query_knowledge(user_input)
            
            if rag_response.get('rag_type') == 'retrieve_and_generate':
                # Successfully generated response
                response = rag_response['generated_response']
                
                # Add metadata about the retrieval
                retrieved_count = rag_response.get('total_retrieved', 0)
                if retrieved_count > 0:
                    response += f"\n\n*Based on {retrieved_count} relevant documents from the knowledge base.*"
                
                return response
                
            elif rag_response.get('rag_type') == 'no_results':
                # No relevant documents found - use LLM with general knowledge
                logger.info("No RAG results found, using LLM with general seismic knowledge")
                return self._handle_no_rag_results(user_input)
                
            else:
                # Error or fallback
                response = rag_response['generated_response']
                # Ensure we never return boolean values
                if isinstance(response, bool):
                    response = str(response)
                return response
                
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            # Fallback to LLM with general knowledge
            return self._handle_no_rag_results(user_input)
    
    def _handle_no_rag_results(self, user_input: str) -> str:
        """
        Handle cases when RAG doesn't find relevant documents by using LLM with general seismic knowledge.
        
        Args:
            user_input: The user's question
            
        Returns:
            str: LLM-generated response using general knowledge
        """
        try:
            # Create a comprehensive system prompt for seismic knowledge
            system_prompt = """You are an expert seismic modeling and geophysics assistant with extensive knowledge of:

**Core Seismic Concepts:**
- Wave propagation physics and properties
- Frequency, bandwidth, and resolution relationships
- Velocity, density, and impedance effects
- Reflection and refraction phenomena
- Attenuation and quality factor (Q)

**Wavelet Theory:**
- Ricker wavelets and their frequency characteristics
- Zero-phase vs minimum-phase wavelets
- Bandwidth and temporal resolution trade-offs
- Source signature design principles

**Forward Modeling:**
- Wedge models and tuning effects
- Thin bed analysis and resolution limits
- Synthetic seismogram generation
- AVO (Amplitude vs Offset) analysis

**Seismic Resolution:**
- Frequency vs resolution relationships
- Tuning thickness and interference effects
- Detection vs resolution limits
- Trade-offs between penetration and resolution

**Rock Physics:**
- Velocity-density relationships
- Fluid effects on seismic properties
- Porosity and permeability impacts
- Lithology identification methods

**Practical Applications:**
- Survey design and acquisition planning
- Processing parameter optimization
- Interpretation workflows
- Reservoir characterization

Guidelines:
1. Provide accurate, educational explanations
2. Use specific examples and typical values when relevant
3. Explain trade-offs and practical considerations
4. Be clear about limitations and uncertainties
5. Structure responses logically with clear sections
6. Include relevant formulas and relationships when helpful

Answer the user's question comprehensively using your knowledge of seismic modeling and geophysics."""

            # Generate response using the LLM
            response = self.llm_client.get_simple_completion(system_prompt, user_input)
            
            # Add a note that this is based on general knowledge
            response += "\n\n*This response is based on general seismic modeling knowledge.*"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Final fallback to basic knowledge base
            return self._fallback_knowledge_response(user_input)
    
    def _fallback_knowledge_response(self, user_input: str) -> str:
        """
        Fallback response when RAG fails.
        
        Args:
            user_input: The user's question
            
        Returns:
            str: Fallback response
        """
        # Try to extract topic from the question
        user_input_lower = user_input.lower()
        
        # Check for specific topics
        if any(word in user_input_lower for word in ['ricker', 'wavelet']):
            response = self.knowledge_base.get_topic_response('ricker', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['wedge', 'model']):
            response = self.knowledge_base.get_topic_response('wedge', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['seismic', 'resolution', 'frequency']):
            response = self.knowledge_base.get_topic_response('seismic_properties', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['rock', 'physics', 'porosity', 'velocity']):
            response = self.knowledge_base.get_topic_response('rock_physics', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        else:
            response = self.knowledge_base.get_topic_response('ricker', 'overview')  # Default topic
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
    
    def _handle_tool_request(self, user_input: str) -> str:
        """
        Handle tool-based requests using the existing tool use pattern.
        
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
                
                # Handle special cases for image outputs
                if self._is_image_output(tool_name, tool_result):
                    return {"image_path": tool_result}
                
                # Handle automatic chaining
                chained_result = self._handle_automatic_chaining(tool_name, tool_input, tool_result)
                if chained_result:
                    return chained_result
                
                # Get final response from LLM
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
                # Ensure we never return boolean values
                if isinstance(result, bool):
                    result = str(result)
                return result
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return f"Error executing tool: {str(e)}"
        else:
            # Return the direct response
            messages.append({"role": "assistant", "content": response["content"]})
            result = self._extract_reply(response["content"]) or response["content"]
            # Ensure we never return boolean values
            if isinstance(result, bool):
                result = str(result)
            return result
    
    def _is_image_output(self, tool_name: str, tool_result: Any) -> bool:
        """
        Check if the tool result is an image output.
        
        Args:
            tool_name: Name of the tool
            tool_result: Result from the tool
            
        Returns:
            bool: True if result is an image path
        """
        return (isinstance(tool_result, str) and 
                tool_result.endswith(".png") and
                tool_name in ["plot_ricker", "plot_wedge_model", "plot_avo_reflectivity", "plot_rock_properties"])
    
    def _handle_automatic_chaining(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any) -> Optional[Dict[str, Any]]:
        """
        Handle automatic chaining of related tools.
        
        Args:
            tool_name: Name of the executed tool
            tool_input: Input parameters for the tool
            tool_result: Result from the tool
            
        Returns:
            Optional dict with image path if chaining occurred
        """
        try:
            # Automatic chaining: if make_ricker, immediately call plot_ricker
            if tool_name == "make_ricker":
                last_wavelet = self.context_manager.get_context("last_ricker_wavelet")
                if last_wavelet:
                    plot_input = {
                        "wavelet": last_wavelet["wavelet"],
                        "time_array": last_wavelet["time_array"]
                    }
                    plot_result = self.tool_manager.process_tool_call("plot_ricker", plot_input)
                    if self._is_image_output("plot_ricker", plot_result):
                        return {"image_path": plot_result}
            
            # Automatic chaining: if wedge_model, immediately call plot_wedge_model
            elif tool_name == "wedge_model":
                last_wedge = self.context_manager.get_context("last_wedge_model")
                if last_wedge and "synthetic" in last_wedge and "parameters" in last_wedge:
                    plot_input = {
                        "synthetic_data": last_wedge["synthetic"],
                        "parameters": last_wedge["parameters"]
                    }
                    plot_result = self.tool_manager.process_tool_call("plot_wedge_model", plot_input)
                    if self._is_image_output("plot_wedge_model", plot_result):
                        return {"image_path": plot_result}
            
            # Automatic chaining: if AVO tools, immediately call plot_avo_reflectivity
            elif tool_name in ["zoeppritz_reflectivity", "shuey_reflectivity"]:
                if isinstance(tool_result, np.ndarray) and "angles" in tool_input:
                    plot_input = {
                        "angles": tool_input["angles"],
                        "rc": tool_result
                    }
                    plot_result = self.tool_manager.process_tool_call("plot_avo_reflectivity", plot_input)
                    if self._is_image_output("plot_avo_reflectivity", plot_result):
                        return {"image_path": plot_result}
            

            
            return None
            
        except Exception as e:
            logger.error(f"Error in automatic chaining: {e}")
            return None

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
                
        except Exception as e:
            logger.error(f"Error updating context: {e}")

    def get_available_tools(self) -> List[Dict]:
        """
        Get list of available tools.
        
        Returns:
            List[Dict]: List of tool schemas
        """
        return self.tools