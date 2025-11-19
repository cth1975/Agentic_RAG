"""Lightweight reranker implementations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

from .embeddings import EmbeddingProvider
from .retrieval import SearchResult


class Reranker(Protocol):
    """Protocol implemented by reranker strategies."""

    def rerank(self, query: str, candidates: Sequence[SearchResult], top_n: int) -> List[SearchResult]:
        ...


@dataclass
class EmbeddingSimilarityReranker:
    """Uses cosine similarity between fresh embeddings to re-order hits."""

    embedding_provider: EmbeddingProvider

    def rerank(self, query: str, candidates: Sequence[SearchResult], top_n: int) -> List[SearchResult]:
        if not candidates:
            return []
        texts = [query] + [candidate.text for candidate in candidates]
        vectors = self.embedding_provider.embed(texts)
        query_vector = vectors[0]
        doc_vectors = vectors[1:]
        scored: List[tuple[float, SearchResult]] = []
        for candidate, vector in zip(candidates, doc_vectors):
            scored.append((_cosine_similarity(query_vector, vector), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        reranked = [candidate for _, candidate in scored[:top_n]]
        return reranked


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def take_top_n(candidates: Iterable[SearchResult], top_n: int) -> List[SearchResult]:
    """Utility to truncate sequences of search results."""

    results: List[SearchResult] = []
    for candidate in candidates:
        results.append(candidate)
        if len(results) >= top_n:
            break
    return results


__all__ = ["Reranker", "EmbeddingSimilarityReranker", "take_top_n"]
