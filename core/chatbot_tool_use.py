import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient
from .tool_manager import ToolManager
from .context_manager import ContextManager
from knowledge.knowledge_base import KnowledgeBase
from core.tool_loop import ToolLoopRunner, extract_reply

logger = logging.getLogger(__name__)

class SeismicChatBotToolUse:
    """
    Seismic ChatBot using the tool use pattern from the notebook.
    This implementation follows the same flow as the example:
    1. System prompt with tool definitions
    2. Tool use flow with proper message handling
    3. Tool execution and result processing
    """
    
    def __init__(self, llm_client=None, tool_manager=None, knowledge_base=None):
        """Initialize the seismic chatbot.

        The LLM client, tool manager, and knowledge base are conversation-stateless
        and expensive to build, so they may be injected (shared across sessions).
        The context manager holds per-conversation state and is ALWAYS fresh, so a
        new instance is fully isolated from any other session. Use ``new_session()``
        to spawn an isolated session that reuses the shared components.
        """
        self.llm_client = llm_client or LLMClient()
        self.tool_manager = tool_manager or ToolManager()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.context_manager = ContextManager()  # per-session, never shared
        self.session_id = uuid.uuid4().hex  # names this session's upload sandbox subdir

        # Get tool schemas for the LLM
        self.tools = self.tool_manager.get_tool_schemas()

        # System prompt following the notebook pattern
        self.system_prompt = self._create_system_prompt()

    def new_session(self) -> "SeismicChatBotToolUse":
        """Return a session-isolated chatbot that shares the heavy, stateless
        components (LLM client, tools, knowledge base) but owns a fresh
        conversation context and token counter."""
        return SeismicChatBotToolUse(
            llm_client=self.llm_client,
            tool_manager=self.tool_manager,
            knowledge_base=self.knowledge_base,
        )

    def attach_image(self, path: str) -> None:
        """Remember the user's uploaded photo (per session) for the outcrop tools."""
        self.context_manager.set_context("last_image", path)

    @property
    def _tool_loop(self) -> ToolLoopRunner:
        """Built fresh on each access from the bot's *current* llm_client /
        tool_manager / context_manager (not cached at __init__ time) so that
        tests which swap those attributes after construction — or construct a
        bare instance via object.__new__ and only set a subset of them — keep
        working unchanged. Cheap: just wraps three references."""
        return ToolLoopRunner(
            getattr(self, "llm_client", None),
            getattr(self, "tool_manager", None),
            getattr(self, "context_manager", None),
        )

    # Tools whose heavy inputs live in per-session context rather than in the
    # LLM's arguments: (tool name, parameter name, context key).
    _CONTEXT_INPUTS = ToolLoopRunner._CONTEXT_INPUTS

    def _inject_context_inputs(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        return self._tool_loop.inject_context_inputs(tool_name, tool_input)

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
- make_ormsby: Creates an Ormsby (bandpass) wavelet from four corner frequencies
- plot_ricker: Plots a Ricker wavelet with time and frequency analysis
- wedge_model: Creates a wedge model for seismic analysis
- plot_wedge_model: Plots wedge model results
- wedge_avo_gather: Builds an AVO angle gather (synthetic wedge per incidence angle) and plots tuning-vs-angle + AVO
- analyze_wedge: Analyzes a wedge model for tuning thickness and amplitude-vs-thickness
- synthetic_seismogram: Builds a general N-layer synthetic seismogram from per-layer thickness (m), Vp, density and optional Vs — reflectivity at each interface (acoustic, or Shuey/Zoeppritz at an incidence angle) convolved with a Ricker/Ormsby wavelet, with a layer-model/reflectivity/trace plot
- zoeppritz_reflectivity: Calculates reflectivity using Zoeppritz equations
- shuey_reflectivity: Calculates reflectivity using Shuey's approximation
- plot_avo_reflectivity: Plots AVO reflectivity curves
- avo_attributes: AVO intercept/gradient + class (I-IV) for an interface, with an intercept-gradient crossplot
- extended_elastic_impedance: Extended Elastic Impedance EEI(χ) for a layer (AI at χ=0), with an EEI-vs-χ plot
- gassmann_substitution: Gassmann fluid substitution from in-situ Vp/Vs/density + porosity (e.g. model the gas case of a brine sand)
- petro_to_avo: End-to-end AVO feasibility from petrophysics — predicts sand & shale elastic properties from porosity/clay, models the AVO response, and returns the intercept/gradient/AVO class with a composite plot.
- fluid_scenario: AVO fluid-substitution scenarios — predicts sand & shale from porosity/clay, Gassmann-substitutes the sand across fluids (e.g. brine vs gas), and returns per-fluid AVO class/intercept/gradient with an overlaid comparison plot (DHI feasibility).
- tuning: Wedge tuning / vertical-resolution analysis — predicts a sand & encasing shale from porosity/clay, builds a sand wedge, and returns the tuning thickness, resolution limit, and amplitude-vs-thickness curve with a plot.
- eei_optimal_chi: Finds the Extended Elastic Impedance rotation angle χ whose EEI log best correlates with a target property log (Vp, Vs, density, and target supplied as logs); returns the optimal χ, the correlation-vs-χ curve, and a plot.
- eei_optimal_chi_petro: EEI optimal-χ from petrophysics — predicts Vp/Vs/density logs from porosity & clay-volume logs, then finds the χ whose EEI best correlates with Vclay (lithology) or porosity, with a plot.
- rock_properties_saturation: Computes Vp, Vs, density, Vp/Vs and impedances at a continuous water saturation Sw from porosity and clay volume, via Gassmann substitution with a Reuss (uniform) or Brie (patchy) brine+hydrocarbon fluid mix.
- saturation_sweep: Sweeps water saturation Sw for one rock (porosity & clay volume) and plots the Vp/Vs/AI saturation curves (the fluid line) under Reuss or Brie mixing — useful for fluid-feasibility / DHI sensitivity.
- run_sweep: Sweep another workflow recipe over a grid of parameter values (cartesian product) and collect one scalar metric per run — returns a results table, summary statistics, a coverage report, and an aggregate plot (line for 1 parameter, heatmap for 2). Use for sensitivity / scenario analysis across ranges of porosity, clay, fluid, saturation, or frequency.
- petro_to_synthetic: N-layer synthetic seismogram from petrophysics — predicts each layer's elastic properties from porosity/clay/fluid (Han 1986 + Gassmann), stacks them with their thicknesses, and returns per-layer properties, interface reflectivities, amplitude metrics, and a layer-model/reflectivity/trace plot.
- interpret_outcrop: Interprets the user's uploaded outcrop photo with a vision model into facies regions (lithology each) plus a scale estimate with confidence, and shows an overlay plot. Use it when a message starts with "[image attached".
- outcrop_to_model: Builds a 2-D elastic earth model from the latest outcrop interpretation on a shale background; takes height_m (overrides the photo's scale; required if none was found) and per-region overrides (lithology / fluid / porosity / vclay keyed by region id or label). Re-run it for corrections — no vision call needed.
- synthetic_section: Convolves the latest 2-D earth model into a synthetic seismic section (wavelet frequency, angle, Shuey/Zoeppritz, time or depth domain) and plots it as an image, wiggle, or both.
- outcrop_to_seismic: One-shot photo → interpretation → 2-D model → seismic section (with both plots). Use when the user uploads a photo and asks directly for the seismic image; use the staged tools when they want to check or correct the interpretation first.

Guidelines:
1. Be helpful and concise in your responses
2. Only use tools when you have all required parameters
3. If you don't have enough information to use a tool correctly, ask follow-up questions
4. For seismic questions, provide educational explanations
5. When using tools, explain what you're doing and interpret the results
6. A user message beginning "[image attached: ...]" means a photo was uploaded this turn: call interpret_outcrop (or outcrop_to_seismic if they ask directly for the seismic response). Never pass image_path, interpretation or model arguments yourself — they are supplied automatically.
7. After interpret_outcrop, report the regions and the scale estimate WITH its confidence, and ask the user to confirm or correct the height before building the model if the confidence is low or no scale was found.

Tool results and plots:
- Tool results are compacted before you see them: long numeric arrays appear as summaries like "<61 values, min=..., max=...>".
- Any plot a tool produces is displayed to the user automatically — never print or mention image file paths.
- Plot tools run automatically after their matching compute tool — never call a plot_* tool yourself, and never pass raw array data as tool arguments.
- After your tools finish, state the key quantitative results (e.g. tuning thickness, AVO class, intercept/gradient, sweep statistics) in your <reply>.

In each conversational turn, you will:
1. Think about the user's request
2. Use tools if needed and you have the required parameters
3. Provide a clear, helpful response

Place all user-facing conversational responses in <reply></reply> XML tags to make them easy to parse.
"""

    def _parse_tool_input(self, tool_input):
        return self._tool_loop.parse_tool_input(tool_input)

    def _compact_tool_result(self, tool_result):
        return self._tool_loop.compact_tool_result(tool_result)

    def _compact_value(self, value):
        return self._tool_loop.compact_value(value)

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

    def process_single_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process a single user input and return a response.

        Args:
            user_input: The user's input text

        Returns:
            dict: {"reply": str, "images": list[str]} — images may be empty.
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

            if isinstance(response, dict) and "reply" in response:
                reply = response["reply"]
                images = list(response.get("images") or [])
            else:
                reply, images = response, []

            # Final safety check: never surface booleans/None as the reply.
            if isinstance(reply, bool):
                reply = str(reply)
            elif reply is None:
                reply = "I didn't get a response. Please try again."
            if not reply and images:
                # An empty final completion would render a blank bubble above
                # the plots; give it a minimal caption instead.
                reply = "Here are the results."

            return {"reply": reply, "images": images}

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"reply": f"I encountered an error: {str(e)}", "images": []}
    
    def _is_knowledge_question(self, user_input: str) -> bool:
        """
        Determine if the user input is a knowledge question using LLM-based intent classification.
        
        Args:
            user_input: The user's input text
            
        Returns:
            bool: True if this should use RAG
        """
        if user_input.lstrip().startswith("[image attached"):
            return False  # an uploaded photo is always a tool request
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

IMPORTANT — this question was NOT matched to the curated knowledge base, so you are
answering from general knowledge only. To avoid misleading the user:
1. Do NOT fabricate or invent specific numeric constants, coefficients, equations, or
   citations. If you are not confident in an exact value, say so rather than making one up.
2. Prefer qualitative explanations and clearly-labelled typical ranges over precise numbers.
3. Explicitly flag uncertainty and recommend authoritative references where appropriate.
4. Provide accurate, educational explanations; structure them logically.

Answer the user's question using your general knowledge of seismic modeling and geophysics,
within the constraints above."""

            # Generate response using the LLM
            response = self.llm_client.get_simple_completion(system_prompt, user_input)

            # Clearly label the answer as NOT grounded in the curated knowledge base.
            disclaimer = (
                "\n\n*⚠️ Not from the curated knowledge base — this is a general-knowledge "
                "answer and may contain inaccuracies. Verify specific values against an "
                "authoritative reference.*"
            )
            return (response + disclaimer).strip()
            
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
    
    def _harvest_images(self, tool_result, collected):
        return self._tool_loop.harvest_images(tool_result, collected)

    def _handle_automatic_chaining(self, tool_name, tool_input, tool_result):
        return self._tool_loop.handle_automatic_chaining(tool_name, tool_input, tool_result)

    def _update_context(self, tool_name, tool_input, tool_result):
        return self._tool_loop.update_context(tool_name, tool_input, tool_result)

    def _extract_reply(self, text):
        return extract_reply(text)

    def _handle_tool_request(self, user_input: str) -> Dict[str, Any]:
        result = self._tool_loop.run(
            self.system_prompt, [{"role": "user", "content": user_input}], self.tools)
        return {"reply": result["reply"], "images": result["images"]}

    def get_available_tools(self) -> List[Dict]:
        """
        Get list of available tools.
        
        Returns:
            List[Dict]: List of tool schemas
        """
        return self.tools