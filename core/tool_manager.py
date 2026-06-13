import logging
from typing import Dict, Any, Tuple
from core.tool_registry import REGISTRY_BY_NAME, TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self):
        # name -> callable, derived from the registry (no drift, no setdefault).
        self.tools = dict(TOOL_FUNCTIONS)
        self.specs = REGISTRY_BY_NAME

    @property
    def tool_configs(self) -> Dict[str, Dict]:
        """Backward-compatibility shim for code that reads tool_configs[name]['required_params'].

        Returns a dict with the same shape as the old hand-maintained AVAILABLE_TOOLS dict,
        derived entirely from the registry so there is no drift or setdefault hacking.
        """
        return {
            name: {
                "required_params": list(spec.required),
                "optional_params": dict(spec.defaults),
            }
            for name, spec in self.specs.items()
        }

    def get_tool_schemas(self) -> list:
        """Return tool schemas wrapped as {"type": "function", "function": {...}}."""
        return [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

    def validate_parameters(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        spec = self.specs.get(tool_name)
        if spec is None:
            return False, f"Unknown tool: {tool_name}"
        for p in spec.required:
            if p not in params:
                return False, f"Missing required parameter: {p}"
        if spec.validator is not None:
            return spec.validator(params)
        return True, ""

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        spec = self.specs.get(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        # Fill defaults BEFORE validating so required-after-default works correctly.
        full_params = dict(spec.defaults)
        full_params.update(params)
        is_valid, msg = self.validate_parameters(tool_name, full_params)
        if not is_valid:
            raise ValueError(msg)
        logger.debug(f"Calling {tool_name} with {full_params}")
        return spec.fn(**full_params)

    def process_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        logger.info(f"Processing tool call: {tool_name} with input: {tool_input}")
        if tool_name not in self.specs:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.execute_tool(tool_name, tool_input)
