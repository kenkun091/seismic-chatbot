import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DATABRICKS_TOKEN, DATABRICKS_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        """Initialize the LLM client with configuration for either DeepSeek or Databricks."""
        # Check if Databricks credentials are available
        if DATABRICKS_TOKEN and DATABRICKS_BASE_URL:
            self.client = OpenAI(
                api_key=DATABRICKS_TOKEN,
                base_url=DATABRICKS_BASE_URL,
            )
        else:
            # Fall back to DeepSeek configuration
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

    def get_completion(self, system_prompt: str, user_prompt: str, tools: Optional[List[Dict]] = None, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get a completion from the LLM with optional tool support.
        
        Args:
            system_prompt: The system prompt to guide the LLM's behavior
            user_prompt: The user's input to process
            tools: List of tool schemas for tool use
            messages: Conversation history for multi-turn conversations
            
        Returns:
            Dict: The LLM's response with content and metadata
        """
        try:
            # Prepare OpenAI-style messages
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                openai_messages.append({"role": "user", "content": user_prompt})
            if messages:  # Only extend if messages is not None
                openai_messages.extend(messages)

            # Prepare parameters for the API call
            api_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Only add tools if they are provided and not None
            if tools:
                api_params["tools"] = tools
            
            response = self.client.chat.completions.create(**api_params)
            
            # Safety check: ensure we have a valid response
            if not response.choices:
                raise ValueError("No choices returned from LLM API")
                
            message = response.choices[0].message
            if not message:
                raise ValueError("No message returned from LLM API")
            # Return a dict compatible with the rest of the code
            result = {
                "content": message.content or "",  # Ensure content is never None
                "tool_calls": getattr(message, "tool_calls", None),
                "stop_reason": getattr(message, "finish_reason", None),
                "usage": getattr(response, "usage", None)
            }
            return result
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def get_simple_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get a simple completion without tool support (backward compatibility).
        
        Args:
            system_prompt: The system prompt to guide the LLM's behavior
            user_prompt: The user's input to process
            
        Returns:
            str: The LLM's response
        """
        response = self.get_completion(system_prompt, user_prompt)
        content = response.get("content", "")
        return content.strip() if content else ""
