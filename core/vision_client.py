"""Vision-capable LLM client for image interpretation.

The chat loop stays on DeepSeek (text-only); this module is used ONLY by
tools/outcrop_tools.py::interpret_outcrop. Two backends behind one protocol:

- AnthropicVisionClient  (anthropic SDK, lazily imported)
- OpenAIVisionClient     (openai SDK pointed at any vision-capable base_url,
                          e.g. OpenAI GPT-4o or a Databricks-served VLM)

`interpret_image` returns the model's raw text; JSON extraction, validation
and the single retry live in the caller, which owns the prompt.
"""
import base64
import logging
from typing import Optional

from typing import Protocol

from config.settings import (VISION_PROVIDER, ANTHROPIC_API_KEY, VISION_API_KEY,
                             VISION_BASE_URL, VISION_MODEL)

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o"}


class VisionClient(Protocol):
    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str: ...


class AnthropicVisionClient:
    def __init__(self, api_key: str, model: Optional[str] = None,
                 max_tokens: int = 2048, client=None):
        self.model = model or DEFAULT_VISION_MODELS["anthropic"]
        self.max_tokens = max_tokens
        self._client = client if client is not None else self._make_sdk_client(api_key)

    @staticmethod
    def _make_sdk_client(api_key: str):
        import anthropic  # lazy: optional dependency
        return anthropic.Anthropic(api_key=api_key)

    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")


class OpenAIVisionClient:
    def __init__(self, api_key: str, base_url: str, model: Optional[str] = None,
                 max_tokens: int = 2048, client=None):
        self.model = model or DEFAULT_VISION_MODELS["openai"]
        self.max_tokens = max_tokens
        self._client = client if client is not None else self._make_sdk_client(api_key, base_url)

    @staticmethod
    def _make_sdk_client(api_key: str, base_url: str):
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)

    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content or ""


def resolve_vision_backend(provider, anthropic_key, vision_key, vision_url) -> Optional[str]:
    """Pick "anthropic" / "openai" / None from configuration (pure; unit-tested).

    Auto mode prefers Anthropic. An explicit provider without its credentials
    raises RuntimeError naming the missing variables.
    """
    p = (provider or "").strip().lower() or None
    if p == "anthropic":
        if not anthropic_key:
            raise RuntimeError("VISION_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        return "anthropic"
    if p == "openai":
        if not (vision_key and vision_url):
            raise RuntimeError(
                "VISION_PROVIDER=openai but VISION_API_KEY and/or VISION_BASE_URL is not set."
            )
        return "openai"
    if p is not None:
        raise RuntimeError(f"Unknown VISION_PROVIDER {provider!r}; use 'anthropic' or 'openai'.")
    if anthropic_key:
        return "anthropic"
    if vision_key and vision_url:
        return "openai"
    return None


def build_vision_client() -> VisionClient:
    """Construct the configured backend, or raise a clear RuntimeError."""
    backend = resolve_vision_backend(VISION_PROVIDER, ANTHROPIC_API_KEY,
                                     VISION_API_KEY, VISION_BASE_URL)
    if backend is None:
        raise RuntimeError(
            "vision provider not configured: set ANTHROPIC_API_KEY (Anthropic) or "
            "VISION_API_KEY + VISION_BASE_URL (OpenAI-compatible) to interpret photos."
        )
    if backend == "anthropic":
        return AnthropicVisionClient(ANTHROPIC_API_KEY, model=VISION_MODEL)
    return OpenAIVisionClient(VISION_API_KEY, VISION_BASE_URL, model=VISION_MODEL)
