"""Typed models shared across ingestion, chunking, and indexing."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import json


@dataclass(slots=True)
class DocumentBlock:
    """Represents a contiguous region of text from the normalized JSON payload."""

    page: int
    kind: str
    text: str


@dataclass(slots=True)
class DocumentRevision:
    """Normalized document payload produced by Phase 0.2."""

    doc_id: str
    rev: str
    title: Optional[str]
    effective_date: Optional[str]
    allowed_groups: List[str]
    source_url: str
    blocks: List[DocumentBlock]
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    hash: Optional[str] = None

    @classmethod
    def from_json(cls, path: Path) -> "DocumentRevision":
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks = [DocumentBlock(**block) for block in payload["blocks"]]
        return cls(
            doc_id=payload["doc_id"],
            rev=payload.get("rev", ""),
            title=payload.get("title"),
            effective_date=payload.get("effective_date"),
            allowed_groups=payload.get("allowed_groups", []),
            source_url=payload.get("source_url", ""),
            blocks=blocks,
            owner=payload.get("owner"),
            tags=payload.get("tags", []),
            hash=payload.get("hash"),
        )


@dataclass(slots=True)
class Chunk:
    """Chunked representation ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    rev: str
    text: str
    page_start: int
    page_end: int
    section_path: Optional[str]
    token_count: int
    allowed_groups: List[str]
    source_url: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedChunk(Chunk):
    """Chunk with embedding vector attached."""

    embedding: List[float] = field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding_created_at: datetime = field(default_factory=datetime.utcnow)


def iter_text_blocks(blocks: Iterable[DocumentBlock], kinds: Optional[set[str]] = None) -> Iterable[DocumentBlock]:
    """Iterate over text-like blocks with optional filtering by kind."""

    for block in blocks:
        if kinds and block.kind not in kinds:
            continue
        if not block.text:
            continue
        yield block


__all__ = [
    "DocumentBlock",
    "DocumentRevision",
    "Chunk",
    "EmbeddedChunk",
    "iter_text_blocks",
]
