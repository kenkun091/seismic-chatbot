import logging
from typing import Dict, Any, Optional
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
        Parse user input using LLM.
        
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
   - TOOL: Which tool to use (make_ricker, plot_ricker, wedge_model)
   - PARAMETERS: Extract specific parameters from the text

Context from previous conversation:
- Last frequency used: {self.context_manager.get_last_frequency()}
- Available in context: {list(self.context_manager.conversation_context.keys())}

Return response as JSON with this structure:
{{
    "intent": "question|action|unclear",
    "tool": "tool_name (if intent=action)",
    "parameters": {{"param1": value1, "param2": value2, ...}},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your analysis"
}}"""

        try:
            llm_response = self.llm_client.get_completion(system_prompt, user_input)
            return self.input_parser.extract_json_from_response(llm_response)
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._fallback_parse(user_input)

    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """
        Fallback parsing when LLM parsing fails.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict[str, Any]: Parsed result using rule-based approach
        """
        intent = self.input_parser.classify_user_intent(user_input)
        
        if intent == "action":
            # Extract frequencies for potential tool parameters
            frequencies = self.input_parser.extract_frequencies(user_input)
            # Extract thickness for wedge model
            thickness = None
            thickness_matches = re.findall(r"(\d+\.?\d*)\s*(?:m|meter|metre|ft|feet|foot|thick|thickness)", user_input.lower())
            if thickness_matches:
                thickness = float(thickness_matches[0])
            # Extract velocities for wedge model
            velocities = self.input_parser.extract_velocities(user_input)
            # Extract densities for wedge model
            densities = self.input_parser.extract_densities(user_input)
            params = {}
            if frequencies:
                params["frequency"] = frequencies[0]
            if thickness:
                params["thickness"] = thickness
            params.update(velocities)
            params.update(densities)
            # Normalize parameters
            params = self.input_parser.normalize_parameters(params)
            # Decide tool
            tool = None
            if "frequency" in params and not "thickness" in params:
                tool = "make_ricker"
            elif "max_thickness" in params:
                tool = "wedge_model"
            if tool:
                return {
                    "intent": "action",
                    "tool": tool,
                    "parameters": params,
                    "confidence": 0.6,
                    "reasoning": "Fallback rule-based parsing with frequency/thickness/velocity/density detection"
                }
        
        return {
            "intent": intent,
            "confidence": 0.5,
            "reasoning": "Fallback rule-based classification"
        }

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
        
        # Normalize parameters to ensure correct mapping
        parameters = self.input_parser.normalize_parameters(parameters)
        
        if not tool_name:
            return "I'm not sure what action you want me to perform. Could you please be more specific?"
            
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
                        'time_length': parameters.get('time_length', 256.0),
                        'amplitude': parameters.get('amplitude', 1.0)
                    })
            # If result is an image path (for plot_ricker), return a special dict
            if tool_name == 'plot_ricker' and isinstance(result, str) and result.endswith('.png'):
                return {'image_path': result, 'parameters': parameters}
            
            return f"Successfully executed {tool_name} with parameters: {parameters}"
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"An error occurred while executing {tool_name}: {str(e)}"

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
