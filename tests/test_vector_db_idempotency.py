"""The vector store must use deterministic, content-derived IDs so repeated
population (within a run and across restarts) upserts in place instead of
accumulating duplicates."""
from knowledge.vector_db import content_id


def test_content_id_is_deterministic():
    assert content_id("hello world", {"domain": "ricker"}) == content_id("hello world", {"domain": "ricker"})


def test_content_id_varies_with_text():
    assert content_id("alpha", {}) != content_id("beta", {})


def test_content_id_varies_with_metadata():
    assert content_id("same text", {"domain": "ricker"}) != content_id("same text", {"domain": "wedge"})


def test_content_id_is_metadata_order_independent():
    assert content_id("x", {"a": 1, "b": 2}) == content_id("x", {"b": 2, "a": 1})


def test_content_id_handles_empty_metadata():
    # None and {} should be treated the same
    assert content_id("x", None) == content_id("x", {})
