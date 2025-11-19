"""Embedding provider abstractions with OpenAI-compatible and mock options."""
from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embeddings for tests and offline development."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeat = (self.dim + len(digest) - 1) // len(digest)
            raw = (digest * repeat)[: self.dim]
            vector = [byte / 255 for byte in raw]
            vectors.append(vector)
        return vectors


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Calls any OpenAI-compatible endpoint using the `/embeddings` contract."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        import httpx  # local import to avoid hard dependency during offline tests

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {api_key}"})
        
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import httpx

        response = self._client.post(
            "/v1/embeddings",
            json={"model": self.model, "input": list(texts)},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    def close(self) -> None:
        self._client.close()


__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
]
