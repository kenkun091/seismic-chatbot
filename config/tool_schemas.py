"""
Backward-compatible re-export. The single source of truth is core.tool_registry.
TOOL_FUNCTIONS here is the name->callable map (was previously dotted strings).
"""
from core.tool_registry import TOOL_SCHEMAS, TOOL_FUNCTIONS  # noqa: F401
