from typing import Dict, Any, Optional

class ContextManager:
    def __init__(self):
        """Initialize the context manager."""
        self.conversation_context: Dict[str, Any] = {}
        self.last_frequency: Optional[float] = None
        self.error_count: int = 0
        self.max_errors: int = 3

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
