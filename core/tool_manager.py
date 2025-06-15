import logging
from typing import Dict, Any, Tuple, Callable
from config.settings import AVAILABLE_TOOLS
from tools.ricker_tools import create_ricker_wavelet, plot_wavelet
from tools.wedge_tools import wedge_model

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self):
        """Initialize the tool manager with available tools."""
        self.tools = {
            'make_ricker': create_ricker_wavelet,
            'plot_ricker': plot_wavelet,
            'wedge_model': wedge_model
        }
        self.tool_configs = AVAILABLE_TOOLS

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
                vp = params.get(f'vp{i}')
                if not vp or vp <= 0:
                    return False, f"Velocity vp{i} must be positive"
                elif vp > 6500 or vp < 1500:
                    return False, f"Invalid vp{i}"
            # Validate densities
            for i in range(1, 4):
                rho = params.get(f'rho{i}')
                if not rho or rho <= 0:
                    return False, f"Density rho{i} must be positive"
        
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
            print(f"DEBUG: Calling {tool_name} with parameters: {full_params}")
            return tool_func(**full_params)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
