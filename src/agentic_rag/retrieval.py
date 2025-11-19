"""Hybrid retrieval helpers for Azure AI Search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import json

from .config import AzureSearchSettings
from .embeddings import EmbeddingProvider


@dataclass(slots=True)
class SearchResult:
    """Normalized shape for retrieved documents."""

    chunk_id: str
    doc_id: str
    rev: str
    text: str
    score: float
    section_path: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    source_url: str
    allowed_groups: Sequence[str]
    metadata: Dict[str, object]
    highlights: Optional[str] = None


class AzureSearchRetriever:
    """Executes hybrid queries (BM25 + vector) with ACL filters."""

    def __init__(
        self,
        settings: AzureSearchSettings,
        embedding_provider: EmbeddingProvider,
        *,
        select_fields: Optional[Sequence[str]] = None,
        semantic_configuration: Optional[str] = None,
        vector_field: str = "embedding_vector",
    ) -> None:
        import httpx

        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.endpoint.rstrip("/"),
            headers={
                "Content-Type": "application/json",
                "api-key": settings.api_key,
            },
            timeout=30,
        )
        self._embedder = embedding_provider
        self._select_fields = select_fields or [
            "id",
            "doc_id",
            "rev",
            "text",
            "section_path",
            "page_start",
            "page_end",
            "source_url",
            "allowed_groups",
            "metadata_json",
        ]
        self._semantic_configuration = semantic_configuration
        self._vector_field = vector_field

    def _filter_for_groups(self, user_groups: Sequence[str]) -> Optional[str]:
        if not user_groups:
            return None
        clauses = [f"allowed_groups/any(g: g eq '{group}')" for group in user_groups]
        return " or ".join(clauses)

    def retrieve(
        self,
        query: str,
        user_groups: Sequence[str],
        *,
        top_k: int,
    ) -> List[SearchResult]:
        vector = self._embedder.embed([query])[0]
        body: Dict[str, object] = {
            "search": query,
            "top": top_k,
            "select": ",".join(self._select_fields),
            "includeTotalResultCount": False,
            "vectors": [
                {
                    "value": vector,
                    "fields": self._vector_field,
                    "k": top_k,
                }
            ],
        }
        filter_clause = self._filter_for_groups(user_groups)
        if filter_clause:
            body["filter"] = filter_clause
        if self._semantic_configuration:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = self._semantic_configuration
        response = self._client.post(
            f"/indexes/{self.settings.index_name}/docs/search?api-version=2023-11-01",
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        results: List[SearchResult] = []
        for raw in payload.get("value", []):
            metadata = raw.get("metadata_json")
            if isinstance(metadata, str) and metadata:
                try:
                    metadata_dict: Dict[str, object] = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata_dict = {"raw": metadata}
            else:
                metadata_dict = {}
            highlight = None
            highlights = raw.get("@search.highlights", {})
            if isinstance(highlights, dict):
                highlight_values = highlights.get("text") or []
                if highlight_values:
                    highlight = " ... ".join(highlight_values)
            results.append(
                SearchResult(
                    chunk_id=raw.get("id", ""),
                    doc_id=raw.get("doc_id", ""),
                    rev=raw.get("rev", ""),
                    text=raw.get("text", ""),
                    score=float(raw.get("@search.score", 0.0)),
                    section_path=raw.get("section_path"),
                    page_start=raw.get("page_start"),
                    page_end=raw.get("page_end"),
                    source_url=raw.get("source_url", ""),
                    allowed_groups=raw.get("allowed_groups", []),
                    metadata=metadata_dict,
                    highlights=highlight,
                )
            )
        return results

    def close(self) -> None:
        self._client.close()


__all__ = ["SearchResult", "AzureSearchRetriever"]
