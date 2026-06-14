"""Minimal, dependency-free security primitives for the network interfaces.

- ``check_api_key``: constant-time comparison of a presented key to the expected key.
- ``RateLimiter``: in-process sliding-window limiter keyed by client identity.

These provide containment (auth + abuse throttling) for the FastAPI/Gradio
entry points, which otherwise proxy straight to a paid LLM endpoint.
"""
import secrets
from collections import defaultdict, deque
from typing import Optional


def check_api_key(provided: Optional[str], expected: str) -> bool:
    """Return True iff ``provided`` matches ``expected`` (constant-time)."""
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


class RateLimiter:
    """Sliding-window rate limiter: at most ``max_requests`` per ``window_seconds`` per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)

    def allow(self, key: str, now: float) -> bool:
        """Record a request at time ``now`` for ``key``; return whether it is allowed."""
        hits = self._hits[key]
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True
