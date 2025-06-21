import logging
from typing import Dict, Any, Optional, List
from .llm_client import LLMClient
from .tool_manager import ToolManager
from .context_manager import ContextManager
from parsing.input_parser import InputParser
from knowledge.knowledge_base import KnowledgeBase
import re

logger = logging.getLogger(__name__)

class SeismicChatBot:
    def __init__(self):
        """Initialize the seismic chatbot with all required components."""
        self.llm_client = LLMClient()
        self.tool_manager = ToolManager()
        self.context_manager = ContextManager()
        self.input_parser = InputParser()
        self.knowledge_base = KnowledgeBase()

    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: The chatbot's response
        """
        try:
            # Parse user input using LLM
            parsed_result = self._parse_user_input(user_input)
            
            # Handle based on intent
            if parsed_result['intent'] == 'question':
                return self._handle_question(user_input)
            elif parsed_result['intent'] == 'action':
                return self._handle_action(parsed_result)
            else:
                return self._handle_unclear_input(user_input)
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self.context_manager.increment_error_count()
            
            if self.context_manager.has_exceeded_max_errors():
                return "I apologize, but I'm having trouble processing your request. Please try rephrasing or ask a different question."
            return f"I encountered an error: {str(e)}"

    def _parse_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input using LLM with enhanced parameter extraction.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict[str, Any]: Parsed result with intent and parameters
        """
        system_prompt = f"""You are an expert seismic modeling assistant. Your job is to analyze user input and determine:

1. INTENT: Classify the user's intent as one of:
   - "question": User is asking for explanations about seismic concepts
   - "action": User wants to execute a seismic modeling tool
   - "unclear": Intent is ambiguous

2. If INTENT is "action", also determine:
   - TOOL: Which tool to use (make_ricker, plot_ricker, wedge_model, zoeppritz_reflectivity, shuey_reflectivity, avo_fluid_indicator)
   - PARAMETERS: Extract specific parameters from the text

**Parameter Extraction Guidelines:**
- For wedge_model: Extract max_thickness, v1, v2, v3, rho1, rho2, rho3
- For make_ricker: Extract frequency, dt, time_length
- For AVO tools: Extract vp1, vs1, rho1, vp2, vs2, rho2, angles, intercept, gradient
- Use synonyms and context (e.g., "depth" for "thickness", "speed" for "velocity", "density" for "rho")
- Use context from previous conversation when parameters are missing
- Convert units appropriately (e.g., "100 meters" → 100, "30 Hz" → 30)
- Handle ranges and arrays (e.g., "velocities [2000, 2500, 3000]" → {{"v1": 2000, "v2": 2500, "v3": 3000}})
- If a parameter is implied but not explicit, make your best guess and mark it as "inferred"
- If you are unsure about a parameter, list it in "missing_parameters" and explain why in "reasoning"

**Context from previous conversation:**
- Last frequency used: {self.context_manager.get_last_frequency()}
- Available in context: {list(self.context_manager.conversation_context.keys())}

Return response as JSON with this structure:
{{
    "intent": "question|action|unclear",
    "tool": "tool_name (if intent=action)",
    "parameters": {{"param1": value1, "param2": value2, ...}},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your analysis, including any inferred or missing parameters",
    "missing_parameters": ["param1", "param2"] (if any required params are missing or uncertain)
}}"""

        try:
            llm_response = self.llm_client.get_completion(system_prompt, user_input)
            parsed_result = self.input_parser.extract_json_from_response(llm_response)
            
            # Validate and enhance the parsed result
            if parsed_result.get('intent') == 'action':
                parsed_result = self._enhance_llm_parameters(parsed_result, user_input)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._fallback_parse(user_input)

    def _enhance_llm_parameters(self, parsed_result: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        Enhance LLM-extracted parameters with rule-based validation, supplementation, and context.
        
        Args:
            parsed_result: Result from LLM parsing
            user_input: Original user input
            
        Returns:
            Dict[str, Any]: Enhanced parsed result
        """
        parameters = parsed_result.get('parameters', {})
        tool = parsed_result.get('tool')
        
        # Use rule-based extraction to supplement LLM results
        if tool == 'wedge_model':
            # Extract velocities and densities using rules
            velocities = self.input_parser.extract_velocities(user_input)
            densities = self.input_parser.extract_densities(user_input)
            
            # Merge with LLM results (LLM takes precedence)
            parameters.update(velocities)
            parameters.update(densities)
            
            # Extract thickness if missing
            if 'max_thickness' not in parameters:
                thickness_matches = re.findall(r"(\d+\.?\d*)\s*(?:m|meter|metre|ft|feet|foot|thick|thickness|depth|deep|layer)", user_input.lower())
                if thickness_matches:
                    parameters['max_thickness'] = float(thickness_matches[0])
            
            # Extract frequency if missing
            if 'wavelet_freq' not in parameters:
                frequencies = self.input_parser.extract_frequencies(user_input)
                if frequencies:
                    parameters['wavelet_freq'] = frequencies[0]
                else:
                    # Use context if available
                    last_freq = self.context_manager.get_last_frequency()
                    if last_freq:
                        parameters['wavelet_freq'] = last_freq
                        parsed_result['reasoning'] = parsed_result.get('reasoning', '') + f" Using last frequency ({last_freq} Hz) from context."
            
            # Use context for missing velocities/densities if available
            last_wedge = self.context_manager.get_context('last_wedge_model')
            if last_wedge and 'parameters' in last_wedge:
                last_params = last_wedge['parameters']
                for param in ['v1', 'v2', 'v3', 'rho1', 'rho2', 'rho3']:
                    if param not in parameters and param in last_params:
                        parameters[param] = last_params[param]
                        parsed_result['reasoning'] = parsed_result.get('reasoning', '') + f" Using {param}={last_params[param]} from previous wedge model."
        
        elif tool == 'make_ricker':
            # Extract frequency if missing
            if 'frequency' not in parameters:
                frequencies = self.input_parser.extract_frequencies(user_input)
                if frequencies:
                    parameters['frequency'] = frequencies[0]
                else:
                    # Use context if available
                    last_freq = self.context_manager.get_last_frequency()
                    if last_freq:
                        parameters['frequency'] = last_freq
                        parsed_result['reasoning'] = parsed_result.get('reasoning', '') + f" Using last frequency ({last_freq} Hz) from context."
            # Force default time_length if not explicitly in user_input
            if 'time_length' not in parameters or not re.search(r'time[_ ]?length|duration|ms|millisecond', user_input, re.IGNORECASE):
                parameters['time_length'] = 256
        
        elif tool == 'plot_ricker':
            # If wavelet/time_array missing, use last_ricker_wavelet from context
            last_wavelet = self.context_manager.get_context('last_ricker_wavelet')
            if last_wavelet:
                if 'wavelet' not in parameters:
                    parameters['wavelet'] = last_wavelet.get('wavelet')
                if 'time_array' not in parameters:
                    parameters['time_array'] = last_wavelet.get('time_array')
                parsed_result['reasoning'] = parsed_result.get('reasoning', '') + " Used last generated wavelet from context."
        
        # Normalize parameters
        parameters = self.input_parser.normalize_parameters(parameters)
        parsed_result['parameters'] = parameters
        
        # Check for missing required parameters
        if tool:
            missing_params = self._check_missing_parameters(tool, parameters)
            if missing_params:
                parsed_result['missing_parameters'] = missing_params
                parsed_result['confidence'] = min(parsed_result.get('confidence', 0.8), 0.7)
        
        return parsed_result

    def _check_missing_parameters(self, tool: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Check for missing required parameters for a given tool.
        
        Args:
            tool: Tool name
            parameters: Extracted parameters
            
        Returns:
            List[str]: List of missing required parameters
        """
        if tool not in self.tool_manager.tool_configs:
            return []
        
        config = self.tool_manager.tool_configs[tool]
        required_params = config.get('required_params', [])
        missing = []
        
        for param in required_params:
            if param not in parameters:
                missing.append(param)
        
        return missing

    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced fallback parsing when LLM parsing fails.
        Uses context, synonyms, and flexible patterns for parameter extraction.
        """
        intent = self.input_parser.classify_user_intent(user_input)
        params = {}
        tool = None
        reasoning = []

        if intent == "action":
            # Extract frequencies with context fallback
            frequencies = self.input_parser.extract_frequencies(user_input)
            if frequencies:
                params["frequency"] = frequencies[0]
                params["wavelet_freq"] = frequencies[0]
                reasoning.append(f"Extracted frequency: {frequencies[0]} Hz")
            else:
                # Use context if available
                last_freq = self.context_manager.get_last_frequency()
                if last_freq:
                    params["frequency"] = last_freq
                    params["wavelet_freq"] = last_freq
                    reasoning.append(f"Using last frequency from context: {last_freq} Hz")

            # Extract thickness with enhanced patterns and synonyms
            thickness = None
            thickness_patterns = [
                r"(\d+\.?\d*)\s*(?:m|meter|metre|ft|feet|foot|thick|thickness|depth|deep|layer)",
                r"thickness\s*[:=]\s*(\d+\.?\d*)",
                r"depth\s*[:=]\s*(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*(?:meters?|metres?|feet?|ft)\s*(?:thick|thickness|depth|deep)"
            ]
            for pattern in thickness_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    thickness = float(match.group(1))
                    break
            if thickness:
                params["thickness"] = thickness
                params["max_thickness"] = thickness
                reasoning.append(f"Extracted thickness: {thickness}")

            # Extract velocities with enhanced patterns
            velocities = self.input_parser.extract_velocities(user_input)
            if velocities:
                params.update(velocities)
                reasoning.append(f"Extracted velocities: {velocities}")
            else:
                # Try alternative velocity patterns
                velocity_patterns = [
                    r"speed\s*[:=]\s*(\d+\.?\d*)",
                    r"velocity\s*[:=]\s*(\d+\.?\d*)",
                    r"(\d+\.?\d*)\s*(?:m/s|mps|meters?/s|metres?/s)"
                ]
                for pattern in velocity_patterns:
                    matches = re.findall(pattern, user_input.lower())
                    if matches:
                        for i, match in enumerate(matches[:3]):
                            params[f'v{i+1}'] = float(match)
                        reasoning.append(f"Extracted velocities using alternative patterns: {[float(m) for m in matches[:3]]}")
                        break

            # Extract densities with enhanced patterns
            densities = self.input_parser.extract_densities(user_input)
            if densities:
                params.update(densities)
                reasoning.append(f"Extracted densities: {densities}")
            else:
                # Try alternative density patterns
                density_patterns = [
                    r"density\s*[:=]\s*(\d+\.?\d*)",
                    r"(\d+\.?\d*)\s*(?:g/cc|g/cm3|g/ccm)"
                ]
                for pattern in density_patterns:
                    matches = re.findall(pattern, user_input.lower())
                    if matches:
                        for i, match in enumerate(matches[:3]):
                            params[f'rho{i+1}'] = float(match)
                        reasoning.append(f"Extracted densities using alternative patterns: {[float(m) for m in matches[:3]]}")
                        break

            # Use context for missing parameters
            context_used = self._apply_context_for_missing_params(params, tool)
            if context_used:
                reasoning.extend(context_used)

            # Normalize parameters
            params = self.input_parser.normalize_parameters(params)

            # Decide tool with enhanced logic
            tool = self._determine_tool_from_params(params, user_input)

            # If plot_ricker, use last_ricker_wavelet from context if needed
            if tool == 'plot_ricker':
                last_wavelet = self.context_manager.get_context('last_ricker_wavelet')
                if last_wavelet:
                    if 'wavelet' not in params:
                        params['wavelet'] = last_wavelet.get('wavelet')
                    if 'time_array' not in params:
                        params['time_array'] = last_wavelet.get('time_array')
                    reasoning.append("Used last generated wavelet from context.")

            # Check for missing required parameters
            missing_params = []
            if tool:
                config = self.tool_manager.tool_configs.get(tool, {})
                required_params = config.get('required_params', [])
                for param in required_params:
                    if param not in params:
                        missing_params.append(param)

            if tool:
                return {
                    "intent": "action",
                    "tool": tool,
                    "parameters": params,
                    "confidence": 0.6,
                    "reasoning": "Enhanced fallback parsing: " + "; ".join(reasoning),
                    "missing_parameters": missing_params
                }

        return {
            "intent": intent,
            "confidence": 0.5,
            "reasoning": "Fallback rule-based classification"
        }

    def _apply_context_for_missing_params(self, params: Dict[str, Any], tool: str) -> List[str]:
        """
        Apply context to fill missing parameters.
        
        Args:
            params: Current parameters
            tool: Tool being used
            
        Returns:
            List[str]: List of reasoning messages about context usage
        """
        reasoning = []
        
        # Use last frequency from context
        if 'frequency' not in params and 'wavelet_freq' not in params:
            last_freq = self.context_manager.get_last_frequency()
            if last_freq:
                params['frequency'] = last_freq
                params['wavelet_freq'] = last_freq
                reasoning.append(f"Using last frequency from context: {last_freq} Hz")
        
        # Use last wedge model parameters
        if tool == 'wedge_model':
            last_wedge = self.context_manager.get_context('last_wedge_model')
            if last_wedge and 'parameters' in last_wedge:
                last_params = last_wedge['parameters']
                for param in ['v1', 'v2', 'v3', 'rho1', 'rho2', 'rho3']:
                    if param not in params and param in last_params:
                        params[param] = last_params[param]
                        reasoning.append(f"Using {param}={last_params[param]} from previous wedge model")
        
        # Use last Ricker wavelet parameters
        elif tool == 'make_ricker':
            last_ricker = self.context_manager.get_context('last_ricker_wavelet')
            if last_ricker:
                for param in ['dt', 'time_length']:
                    if param not in params and param in last_ricker:
                        params[param] = last_ricker[param]
                        reasoning.append(f"Using {param}={last_ricker[param]} from previous Ricker wavelet")
        
        return reasoning

    def _determine_tool_from_params(self, params: Dict[str, Any], user_input: str) -> str:
        """
        Determine the appropriate tool based on parameters and user input.
        
        Args:
            params: Extracted parameters
            user_input: Original user input
            
        Returns:
            str: Tool name
        """
        text = user_input.lower()
        
        # Check for explicit tool mentions
        if any(word in text for word in ['ricker', 'wavelet']):
            if any(word in text for word in ['plot', 'show', 'display', 'visualize']):
                return 'plot_ricker'
            else:
                return 'make_ricker'
        
        # Enhanced logic for wedge model plotting
        if any(word in text for word in ['wedge', 'model', 'tuning']):
            if any(word in text for word in ['plot', 'show', 'display', 'visualize']):
                return 'plot_wedge_model'
            else:
                return 'wedge_model'
        
        if any(word in text for word in ['avo', 'reflectivity', 'zoeppritz']):
            return 'zoeppritz_reflectivity'
        
        if any(word in text for word in ['shuey', 'approximation']):
            return 'shuey_reflectivity'
        
        if any(word in text for word in ['fluid', 'indicator']):
            return 'avo_fluid_indicator'
        
        # Infer from parameters
        if 'frequency' in params and 'max_thickness' not in params:
            return 'make_ricker'
        elif 'max_thickness' in params:
            return 'wedge_model'
        elif 'vp1' in params and 'vp2' in params:
            return 'zoeppritz_reflectivity'
        elif 'intercept' in params and 'gradient' in params:
            return 'avo_fluid_indicator'
        
        return None

    def _handle_question(self, user_input: str) -> str:
        """
        Handle question-type inputs.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Response to the question
        """
        # Determine topic and subtopic from input
        text = user_input.lower()
        
        if any(word in text for word in ['ricker', 'wavelet']):
            if any(word in text for word in ['frequency', 'hz', 'hertz', 'bandwidth']):
                return self.knowledge_base.get_topic_response('ricker', 'frequency')
            elif any(word in text for word in ['create', 'make', 'generate', 'build']):
                return self.knowledge_base.get_topic_response('ricker', 'creation')
            else:
                return self.knowledge_base.get_topic_response('ricker')
                
        elif any(word in text for word in ['wedge', 'model', 'modeling', 'tuning']):
            return self.knowledge_base.get_topic_response('wedge')
            
        elif any(word in text for word in ['seismic', 'properties', 'velocity', 'density']):
            return self.knowledge_base.get_topic_response('seismic_properties')
            
        else:
            return self.knowledge_base.get_topic_response('overview')

    def _handle_action(self, parsed_result: Dict[str, Any]) -> str:
        """
        Handle action-type inputs.
        
        Args:
            parsed_result: Parsed result from input parsing
            
        Returns:
            str: Response to the action
        """
        tool_name = parsed_result.get('tool')
        parameters = parsed_result.get('parameters', {})
        missing_params = parsed_result.get('missing_parameters', [])
        confidence = parsed_result.get('confidence', 0.5)
        
        # Normalize parameters to ensure correct mapping
        parameters = self.input_parser.normalize_parameters(parameters)
        
        if not tool_name:
            return "I'm not sure what action you want me to perform. Could you please be more specific?"
        
        # Handle missing parameters
        if missing_params:
            return self._handle_missing_parameters(tool_name, missing_params, parameters, confidence)
            
        try:
            # If plotting and wavelet not provided, try to fetch from context
            if tool_name == 'plot_ricker' and 'wavelet' not in parameters:
                last_wavelet = self.context_manager.get_context('last_ricker_wavelet')
                if last_wavelet:
                    parameters['time_array'] = last_wavelet['time_array']
                    parameters['wavelet'] = last_wavelet['wavelet']
                else:
                    return "Error: No previously generated wavelet found. Please create a Ricker wavelet first."

            result = self.tool_manager.execute_tool(tool_name, parameters)
            
            # Store wavelet in context if created
            if tool_name == 'make_ricker' and 'frequency' in parameters:
                self.context_manager.update_frequency(parameters['frequency'])
                # Store both time array and wavelet
                if isinstance(result, tuple) and len(result) == 2:
                    self.context_manager.update_context('last_ricker_wavelet', {
                        'time_array': result[0],
                        'wavelet': result[1],
                        'frequency': parameters['frequency'],
                        'dt': parameters.get('dt', 0.001),
                        'time_length': parameters.get('time_length', 256),
                        'amplitude': parameters.get('amplitude', 1.0)
                    })
            
            # Store wedge model results in context
            if tool_name == 'wedge_model' and isinstance(result, tuple) and len(result) == 4:
                self.context_manager.update_context('last_wedge_model', {
                    'time_array': result[0],
                    'model': result[1],
                    'synthetic_data': result[2],
                    'parameters': result[3]
                })
                # Now, call the plotting tool
                plot_result = self.tool_manager.execute_tool('plot_wedge_model', {
                    'synthetic_data': result[2],
                    'parameters': result[3]
                })
                return {'image_path': plot_result, 'parameters': parameters}

            # If result is an image path (for plot_ricker), return a special dict
            if tool_name == 'plot_ricker' and isinstance(result, str) and result.endswith('.png'):
                return {'image_path': result, 'parameters': parameters}
            
            return f"Successfully executed {tool_name} with parameters: {parameters}"
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"An error occurred while executing {tool_name}: {str(e)}"

    def _handle_missing_parameters(self, tool_name: str, missing_params: List[str], 
                                 current_params: Dict[str, Any], confidence: float) -> str:
        """
        Handle missing parameters by suggesting defaults or asking for clarification.
        
        Args:
            tool_name: Name of the tool
            missing_params: List of missing parameters
            current_params: Currently extracted parameters
            confidence: Confidence level of the parsing
            
        Returns:
            str: Response asking for missing parameters or suggesting defaults
        """
        # Define default values for common parameters
        defaults = {
            'wedge_model': {
                'v1': 2000, 'v2': 2500, 'v3': 3000,
                'rho1': 2.1, 'rho2': 2.2, 'rho3': 2.3,
                'wavelet_freq': 30, 'max_thickness': 100,
                'num_traces': 61, 'dt': 0.1, 'wavelet_length': 500.0,
                'phase_rot': 0.0, 'wv_type': 'ricker', 'gain': 1.0,
                'plotpadtime': 50.0, 'thickness_domain': 'depth', 'zunit': 'm'
            },
            'make_ricker': {
                'frequency': 30, 'dt': 0.001, 'time_length': 256
            }
        }
        
        tool_defaults = defaults.get(tool_name, {})
        
        # Check if we can use defaults for all missing parameters
        can_use_defaults = all(param in tool_defaults for param in missing_params)
        
        if can_use_defaults and confidence > 0.6:
            # Use defaults and execute
            for param in missing_params:
                current_params[param] = tool_defaults[param]
            
            try:
                result = self.tool_manager.execute_tool(tool_name, current_params)
                used_defaults = ", ".join([f"{param}={tool_defaults[param]}" for param in missing_params])
                
                # Store results in context
                if tool_name == 'wedge_model' and isinstance(result, tuple) and len(result) == 4:
                    self.context_manager.update_context('last_wedge_model', {
                        'time_array': result[0],
                        'model': result[1],
                        'synthetic_data': result[2],
                        'parameters': result[3]
                    })
                    
                    # Now, call the plotting tool
                    plot_result = self.tool_manager.execute_tool('plot_wedge_model', {
                        'synthetic_data': result[2],
                        'parameters': result[3]
                    })
                    
                    # Return image path and success message
                    return {
                        'image_path': plot_result,
                        'message': f"""Successfully executed {tool_name} using default values for missing parameters: {used_defaults}
Full parameters used: {str(current_params)}
The model has been created and stored in context for future reference."""
                    }

                if tool_name == 'make_ricker' and isinstance(result, tuple) and len(result) == 2:
                    self.context_manager.update_frequency(current_params['frequency'])
                    self.context_manager.update_context('last_ricker_wavelet', {
                        'time_array': result[0],
                        'wavelet': result[1],
                        'frequency': current_params['frequency'],
                        'dt': current_params.get('dt', 0.001),
                        'time_length': current_params.get('time_length', 256)
                    })

                return f"""Successfully executed {tool_name} using default values for missing parameters: {used_defaults}

Full parameters used: {str(current_params)}

The model has been created and stored in context for future reference."""
                
            except Exception as e:
                return f"Error using default parameters: {str(e)}"
        else:
            # Ask user for missing parameters with helpful suggestions
            param_descriptions = {
                'max_thickness': 'maximum thickness of the wedge (e.g., 100 meters)',
                'v1': 'velocity of layer 1 (e.g., 2000 m/s)',
                'v2': 'velocity of layer 2 (e.g., 2500 m/s)',
                'v3': 'velocity of layer 3 (e.g., 3000 m/s)',
                'rho1': 'density of layer 1 (e.g., 2.3 g/cc)',
                'rho2': 'density of layer 2 (e.g., 2.2 g/cc)',
                'rho3': 'density of layer 3 (e.g., 2.3 g/cc)',
                'frequency': 'wavelet frequency (e.g., 30 Hz)',
                'wavelet_freq': 'wavelet frequency (e.g., 30 Hz)',
                'num_traces': 'number of traces in the model (e.g., 61)',
                'dt': 'time sampling interval (e.g., 0.1 ms)',
                'wavelet_length': 'length of the wavelet (e.g., 512 ms)',
                'phase_rot': 'phase rotation in degrees (e.g., 0)',
                'wv_type': 'wavelet type (e.g., "ricker" or "ormsby")',
                'gain': 'gain factor for display (e.g., 1.0)',
                'plotpadtime': 'padding time for plots (e.g., 50 ms)',
                'thickness_domain': 'domain for thickness (e.g., "depth" or "time")',
                'zunit': 'unit for depth/thickness (e.g., "m" or "ft")'
            }
            
            missing_descriptions = [param_descriptions.get(param, param) for param in missing_params]
            missing_text = ", ".join(missing_descriptions)
            
            # Provide example commands
            examples = self._get_example_commands(tool_name, missing_params)
            
            return f"""I need some additional information to execute {tool_name}. 

**Missing parameters:** {missing_text}

**Current parameters:** {str(current_params)}

**You can:**
1. Provide the missing values explicitly
2. Use reasonable defaults (I can suggest these)
3. Use values from a previous model in context

**Example commands:**
{examples}

**Or simply say "use defaults" and I'll fill in reasonable values."""
    
    def _get_example_commands(self, tool_name: str, missing_params: List[str]) -> str:
        """
        Generate example commands for missing parameters.
        
        Args:
            tool_name: Name of the tool
            missing_params: List of missing parameters
            
        Returns:
            str: Example commands
        """
        if tool_name == 'wedge_model':
            examples = []
            if 'max_thickness' in missing_params:
                examples.append('- "with 100m thickness"')
            if any(p in missing_params for p in ['v1', 'v2', 'v3']):
                examples.append('- "velocities [2000, 2500, 3000]"')
            if any(p in missing_params for p in ['rho1', 'rho2', 'rho3']):
                examples.append('- "densities [2.1, 2.2, 2.3]"')
            if 'wavelet_freq' in missing_params:
                examples.append('- "30 Hz wavelet"')
            return "\n".join(examples)
        
        elif tool_name == 'make_ricker':
            examples = []
            if 'frequency' in missing_params:
                examples.append('- "30 Hz Ricker wavelet"')
            if 'dt' in missing_params:
                examples.append('- "with 0.001s sampling"')
            return "\n".join(examples)
        
        return "- Provide the missing parameters in any format you prefer"

    def _handle_unclear_input(self, user_input: str) -> str:
        """
        Handle unclear inputs.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Response for unclear input
        """
        return """I'm not sure what you're asking for. Could you please:

1. Ask a specific question about seismic modeling, or
2. Request a specific action (like creating a Ricker wavelet or wedge model)

For example:
- "What is a Ricker wavelet?"
- "Create a 30 Hz Ricker wavelet"
- "Explain tuning effects in wedge models"
- "Make a wedge model with 100m thickness"
"""
