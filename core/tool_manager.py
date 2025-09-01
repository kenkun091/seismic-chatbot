import logging
import importlib
from typing import Dict, Any, Tuple, Callable
from config.settings import AVAILABLE_TOOLS
from config.tool_schemas import TOOL_SCHEMAS, TOOL_FUNCTIONS
from tools.ricker_tools import create_ricker_wavelet, plot_wavelet
from tools.wedge_tools import create_wedge_model, plot_wedge_model
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity
from tools.rock_physics_tools import calculate_rock_properties, rock_physics_rag
from tools.rag_tools import knowledge_rag

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self):
        """Initialize the tool manager with available tools."""
        # Direct function mapping for backward compatibility
        self.tools = {
            'make_ricker': create_ricker_wavelet,
            'plot_ricker': plot_wavelet,
            'wedge_model': create_wedge_model,
            'plot_wedge_model': plot_wedge_model,
            'zoeppritz_reflectivity': zoeppritz_reflectivity,
            'shuey_reflectivity': shuey_reflectivity,
            'plot_avo_reflectivity': plot_avo_reflectivity,
            'calculate_rock_properties': calculate_rock_properties,
            'rock_physics_rag': rock_physics_rag,
            'knowledge_rag': knowledge_rag
        }
        self.tool_configs = AVAILABLE_TOOLS
        self.tool_schemas = {tool["name"]: tool for tool in TOOL_SCHEMAS}
        
        # Add configs for new tools if not present
        self.tool_configs.setdefault('zoeppritz_reflectivity', {
            'required_params': ['vp1', 'vs1', 'rho1', 'vp2', 'vs2', 'rho2', 'angles'],
            'optional_params': {}
        })
        self.tool_configs.setdefault('shuey_reflectivity', {
            'required_params': ['vp1', 'vs1', 'rho1', 'vp2', 'vs2', 'rho2', 'angles'],
            'optional_params': {}
        })

        self.tool_configs.setdefault('plot_wedge_model', {
            'required_params': ['synthetic_data', 'parameters'],
            'optional_params': {}
        })
        self.tool_configs.setdefault('plot_avo_reflectivity', {
            'required_params': ['angles', 'rc'],
            'optional_params': {}
        })
        
        self.tool_configs.setdefault('calculate_rock_properties', {
            'required_params': ['phit', 'vclay'],
            'optional_params': {'fluid_type': 'water'}
        })
        

        
        self.tool_configs.setdefault('rock_physics_rag', {
            'required_params': ['query'],
            'optional_params': {'top_k': 3}
        })
        
        self.tool_configs.setdefault('knowledge_rag', {
            'required_params': ['query'],
            'optional_params': {'domain': None, 'top_k': 3}
        })

    def get_tool_schemas(self) -> list:
        """
        Get the list of tool schemas for the LLM in OpenAI/DeepSeek format.
        
        Returns:
            list: List of tool schemas
        """
        # Wrap each tool as {"type": "function", "function": {...}}
        formatted_tools = []
        for tool in self.tool_schemas.values():
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return formatted_tools

    def process_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Process a tool call with the given parameters.
        This is the main function for handling tool calls from the LLM.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of parameters for the tool
            
        Returns:
            Any: The result of the tool execution
        """
        logger.info(f"Processing tool call: {tool_name} with input: {tool_input}")
        
        try:
            # Validate parameters
            is_valid, error_message = self.validate_parameters(tool_name, tool_input)
            if not is_valid:
                raise ValueError(error_message)
            
            # Execute the tool
            return self.execute_tool(tool_name, tool_input)
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            raise

    def validate_parameters(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters for a specific tool.
        
        Args:
            tool_name: Name of the tool to validate parameters for
            params: Dictionary of parameters to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if tool_name not in self.tools:
            return False, f"Unknown tool: {tool_name}"

        config = self.tool_configs[tool_name]
        
        # Check required parameters
        for param in config['required_params']:
            if param not in params:
                return False, f"Missing required parameter: {param}"

        # Validate specific tool parameters
        if tool_name == 'make_ricker':
            freq = params.get('frequency')
            if not freq or freq <= 0 or freq > 1000:
                return False, "Frequency must be between 0 and 1000 Hz"
            
            dt = params.get('dt', 0.001)
            if dt <= 0 or dt > 0.1:
                return False, "Sampling interval (dt) must be between 0 and 0.1 seconds"
                
        elif tool_name == 'wedge_model':
            thickness = params.get('max_thickness')
            if not thickness or thickness <= 0:
                return False, "Maximum thickness must be positive"
            
            # Validate velocities
            for i in range(1, 4):
                v = params.get(f'v{i}')
                if not v or v <= 0:
                    return False, f"Velocity v{i} must be positive"
                elif v > 6500 or v < 1500:
                    return False, f"Invalid v{i}"
            # Validate densities
            for i in range(1, 4):
                rho = params.get(f'rho{i}')
                if not rho or rho <= 0:
                    return False, f"Density rho{i} must be positive"
        elif tool_name == 'zoeppritz_reflectivity' or tool_name == 'shuey_reflectivity':
            for param in ['vp1', 'vs1', 'rho1', 'vp2', 'vs2', 'rho2', 'angles']:
                if param not in params:
                    return False, f"Missing required parameter: {param}"

        
        return True, ""

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Dictionary of parameters for the tool
            
        Returns:
            Any: The result of the tool execution
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Fill in missing optional parameters with defaults
        config = self.tool_configs[tool_name]
        full_params = params.copy()
        # Add optional params if missing
        for k, v in config.get('optional_params', {}).items():
            if k not in full_params:
                full_params[k] = v
        # Validate parameters
        is_valid, error_message = self.validate_parameters(tool_name, full_params)
        if not is_valid:
            raise ValueError(error_message)

        try:
            # Execute the tool
            tool_func = self.tools[tool_name]
            logger.debug(f"Calling {tool_name} with parameters: {full_params}")
            return tool_func(**full_params)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
