from typing import Dict, Any, Optional

from core.turn_trace import TraceRecorder

class ContextManager:
    def __init__(self):
        """Initialize the context manager."""
        self.conversation_context: Dict[str, Any] = {}
        self.last_frequency: Optional[float] = None
        self.error_count: int = 0
        self.max_errors: int = 3
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # Per-session decision-trace recorder (core/turn_trace.py). The owning
        # bot stamps its session_id onto it right after construction.
        self.trace = TraceRecorder()

    def update_context(self, key: str, value: Any) -> None:
        """
        Update a specific context value.
        
        Args:
            key: The context key to update
            value: The new value to store
        """
        self.conversation_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context.
        
        Args:
            key: The context key to retrieve
            default: Default value if key not found
            
        Returns:
            Any: The context value or default
        """
        return self.conversation_context.get(key, default)

    def update_frequency(self, frequency: float) -> None:
        """
        Update the last used frequency.
        
        Args:
            frequency: The frequency value to store
        """
        self.last_frequency = frequency

    def get_last_frequency(self) -> Optional[float]:
        """
        Get the last used frequency.
        
        Returns:
            Optional[float]: The last frequency or None
        """
        return self.last_frequency

    def increment_error_count(self) -> None:
        """Increment the error counter."""
        self.error_count += 1

    def reset_error_count(self) -> None:
        """Reset the error counter to zero."""
        self.error_count = 0

    def has_exceeded_max_errors(self) -> bool:
        """
        Check if maximum error count has been exceeded.
        
        Returns:
            bool: True if max errors exceeded
        """
        return self.error_count >= self.max_errors

    def clear_context(self) -> None:
        """Clear all context data."""
        self.conversation_context.clear()
        self.last_frequency = None
        self.error_count = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def set_context(self, key: str, value: Any) -> None:
        """
        Alias for update_context for compatibility.
        """
        self.update_context(key, value)
        
    def update_token_usage(self, usage) -> None:
        """
        Update token usage statistics.
        
        Args:
            usage: Object containing token usage information
        """
        if usage:
            # Handle both dictionary and CompletionUsage object
            try:
                # Try dictionary access first
                if hasattr(usage, "get"):
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                # Then try object attribute access
                else:
                    self.token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
                    self.token_usage["total_tokens"] += getattr(usage, "total_tokens", 0)
            except Exception as e:
                # Log error but don't crash
                print(f"Error updating token usage: {e}")
                pass
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get the current token usage statistics.
        
        Returns:
            Dict[str, int]: Dictionary with token usage counts
        """
        return self.token_usage
