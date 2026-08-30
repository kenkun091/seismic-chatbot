"""Semantic index over the tool registry — the discovery half of agentic mode.

Derived entirely from core.tool_registry.REGISTRY: each compute tool renders to
one 'card' embedded into a dedicated ChromaDB collection. Auto-plot TARGETS are
excluded (the LLM must never call plot_* directly; chaining handles plots).
Population is idempotent (content-derived IDs) and self-cleaning (stored IDs no
longer derivable from the registry are deleted, so renamed/removed tools cannot
linger).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import RAG_EMBEDDING_MODEL, RAG_VECTOR_DB_PATH
from core.tool_registry import REGISTRY
from knowledge.vector_db import content_id

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tool_index"


def render_card(spec) -> str:
    parts = []
    for pname, meta in spec.params.items():
        required = ", required" if pname in spec.required else ""
        parts.append(f"{pname} ({meta.get('type', 'any')}{required})")
    card = f"{spec.name}: {spec.description}"
    if parts:
        card += " Params: " + ", ".join(parts)
    return card


@dataclass(frozen=True)
class ToolCard:
    name: str
    card: str
    required: tuple
    score: float


class ToolIndex:
    def __init__(self, persist_directory: str | None = None, specs: list | None = None):
        specs = REGISTRY if specs is None else specs
        plot_targets = {s.auto_plot for s in specs if s.auto_plot}
        self._specs = [s for s in specs if s.name not in plot_targets]
        self.client = chromadb.PersistentClient(path=persist_directory or RAG_VECTOR_DB_PATH)
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(
                COLLECTION_NAME,
                metadata={"description": "LLM-facing tool cards", "hnsw:space": "cosine"})
        self.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        self._populate()

    def _populate(self) -> None:
        cards = {content_id(render_card(s), {"tool": s.name}): s for s in self._specs}
        stored = set(self.collection.get().get("ids", []))
        stale = sorted(stored - set(cards))
        if stale:
            self.collection.delete(ids=stale)
            logger.info(f"tool_index: deleted {len(stale)} stale cards")
        new_ids = [i for i in cards if i not in stored]
        if not new_ids:
            return
        texts = [render_card(cards[i]) for i in new_ids]
        metas = [{"tool": cards[i].name, "required": ",".join(cards[i].required)}
                 for i in new_ids]
        embeddings = self.embedding_model.encode(texts).tolist()
        self.collection.upsert(ids=new_ids, documents=texts,
                               metadatas=metas, embeddings=embeddings)
        logger.info(f"tool_index: embedded {len(new_ids)} cards")

    def search(self, task_description: str, top_k: int = 5,
               threshold: float = 0.2) -> list[ToolCard]:
        """Top-k cards by cosine similarity. The top 3 are ALWAYS returned
        (an on-topic request must never get an empty discovery); results
        beyond the top 3 must clear `threshold`."""
        n = min(top_k, self.collection.count())
        if n == 0:
            return []
        query_embedding = self.embedding_model.encode(task_description).tolist()
        res = self.collection.query(query_embeddings=[query_embedding], n_results=n,
                                    include=["documents", "metadatas", "distances"])
        cards = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0]):
            score = 1.0 - dist  # cosine space
            if len(cards) >= 3 and score < threshold:
                continue
            required = tuple(p for p in meta.get("required", "").split(",") if p)
            cards.append(ToolCard(name=meta["tool"], card=doc,
                                  required=required, score=score))
        return cards
