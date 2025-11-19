"""Indexing helpers (Azure AI Search friendly)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import EmbeddedChunk


class AzureSearchIndexer:
    """Minimal client to push chunks into Azure AI Search."""

    def __init__(self, endpoint: str, api_key: str, index_name: str) -> None:
        import httpx

        self.endpoint = endpoint.rstrip("/")
        self.index_name = index_name
        self._client = httpx.Client(
            base_url=self.endpoint,
            headers={"Content-Type": "application/json", "api-key": api_key},
        )

    def upload(self, chunks: Iterable[EmbeddedChunk]) -> None:
        import httpx

        actions = [
            {
                "@search.action": "mergeOrUpload",
                "id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "rev": chunk.rev,
                "text": chunk.text,
                "embedding_vector": chunk.embedding,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "allowed_groups": chunk.allowed_groups,
                "source_url": chunk.source_url,
                "metadata_json": json.dumps(chunk.metadata),
                "embedding_model": chunk.embedding_model,
                "embedding_created_at": chunk.embedding_created_at.isoformat(),
            }
            for chunk in chunks
        ]
        response = self._client.post(f"/indexes/{self.index_name}/docs/index?api-version=2023-11-01", json={"value": actions})
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()


class JSONLIndexer:
    """Writes chunks to disk for offline inspection or handoff to other systems."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def upload(self, chunks: Iterable[EmbeddedChunk]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                record = {
                    "id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "rev": chunk.rev,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                    "meta": chunk.metadata,
                }
                f.write(json.dumps(record) + "\n")


__all__ = ["AzureSearchIndexer", "JSONLIndexer"]
