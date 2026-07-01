"""Retrieval result type + default top-K/N knobs for the sqlite-vec store.

The chatbot issues its own dense top-K query against ``VectorStore`` and
builds ``RetrievedRecord``s directly (see ``chatbot_dispatch``). This module
is the shared home for the ``RetrievedRecord`` result type and the default
top-K / top-N knobs.

The generative-LLM rerank step that once lived here — its prompt builder,
its ``complete()`` call, and the candidate-line formatter — was dropped
pre-cutover; the live path is dense-retrieval-only and dense distance order
is the result order.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.rag_vector_store import StoredRecord, VectorStore


# Configurable knobs. ``DEFAULT_TOP_K`` is what we ask the vector store for;
# ``DEFAULT_TOP_N`` is the cap the answer composer renders. The 30 → 10 funnel
# keeps the candidate pool wide while bounding the synthesis prompt; callers
# may override both.
DEFAULT_TOP_K = 30
DEFAULT_TOP_N = 10


@dataclass(frozen=True)
class RetrievedRecord:
    """One result row: the source record, its dense-retrieval distance, and
    an optional rerank score (None for dense-only retrieval; reserved for a
    future cross-encoder reranker).
    """
    record: StoredRecord
    distance: float
    rerank_score: float | None


__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_N",
    "RetrievedRecord",
    "VectorStore",  # re-export for RAG caller ergonomics
]
