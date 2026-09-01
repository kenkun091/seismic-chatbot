import logging
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DATABRICKS_TOKEN, DATABRICKS_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from core.turn_trace import emit_event, usage_dict

logger = logging.getLogger(__name__)


def resolve_llm_credentials(deepseek_key, deepseek_url, databricks_token, databricks_url):
    """Select an LLM provider's (api_key, base_url) or fail fast with a clear message.

    Databricks takes precedence when both its credentials are present; otherwise
    DeepSeek is used. Raises ``RuntimeError`` if no provider is fully configured,
    so the app fails at startup instead of constructing a client with a None key
    and erroring opaquely on the first request.
    """
    if databricks_token and databricks_url:
        return databricks_token, databricks_url
    if deepseek_key and deepseek_url:
        return deepseek_key, deepseek_url
    if deepseek_key and not deepseek_url:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is set but DEEPSEEK_BASE_URL is missing. "
            "Set both, or provide DATABRICKS_TOKEN + DATABRICKS_BASE_URL."
        )
    raise RuntimeError(
        "No LLM credentials found. Set DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL "
        "(or DATABRICKS_TOKEN + DATABRICKS_BASE_URL) in your environment / .env."
    )


class LLMClient:
    def __init__(self):
        """Initialize the LLM client with configuration for either DeepSeek or Databricks."""
        api_key, base_url = resolve_llm_credentials(
            DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DATABRICKS_TOKEN, DATABRICKS_BASE_URL
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
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

            start = time.perf_counter()
            response = self.client.chat.completions.create(**api_params)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            
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
                "usage": getattr(response, "usage", None),
                "model": self.model,
                "latency_ms": latency_ms
            }
            return result
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def get_simple_completion(self, system_prompt: str, user_prompt: str,
                              context_manager=None) -> str:
        """Text-only completion. When a context_manager is supplied, its token
        counter and decision trace are updated — this is how KnowledgeRouter
        calls (intent classification, no-RAG fallback) become accountable."""
        response = self.get_completion(system_prompt, user_prompt)
        if context_manager is not None:
            if response.get("usage"):
                context_manager.update_token_usage(response["usage"])
            emit_event(context_manager, "llm",
                       model=response.get("model"),
                       latency_ms=response.get("latency_ms"),
                       **usage_dict(response.get("usage")))
        content = response.get("content", "")
        return content.strip() if content else ""
