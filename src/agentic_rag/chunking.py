"""Heading-aware chunking utilities aligned with the implementation plan."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .models import Chunk, DocumentRevision, DocumentBlock


def _estimate_tokens(text: str) -> int:
    """Rough token estimator (word count * 1.3)."""

    words = len(text.split())
    return max(1, int(words * 1.3))


@dataclass(slots=True)
class ChunkingConfig:
    target_tokens: int = 500
    overlap_tokens: int = 75
    max_tokens: int = 800


def chunk_document(doc: DocumentRevision, config: ChunkingConfig | None = None) -> List[Chunk]:
    """Chunk a normalized document into overlapping passages."""

    config = config or ChunkingConfig()
    chunks: List[Chunk] = []
    buffer: List[DocumentBlock] = []
    buffer_tokens = 0
    chunk_index = 0

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_tokens, chunk_index
        if not buffer:
            return
        if not force and buffer_tokens < config.target_tokens:
            return
        text = "\n".join(block.text for block in buffer)
        token_count = _estimate_tokens(text)
        chunk_id = f"{doc.doc_id}:{doc.rev}:{chunk_index:04d}"
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            rev=doc.rev,
            text=text,
            page_start=min(block.page for block in buffer),
            page_end=max(block.page for block in buffer),
            section_path=_build_section_path(buffer),
            token_count=token_count,
            allowed_groups=doc.allowed_groups,
            source_url=doc.source_url,
            metadata={"chunk_index": chunk_index, "title": doc.title},
        )
        chunks.append(chunk)
        chunk_index += 1
        # apply overlap
        buffer = _tail_within_tokens(buffer, config.overlap_tokens)
        buffer_tokens = sum(_estimate_tokens(block.text) for block in buffer)

    for block in doc.blocks:
        buffer.append(block)
        buffer_tokens += _estimate_tokens(block.text)
        if buffer_tokens >= config.target_tokens:
            flush()
        if buffer_tokens >= config.max_tokens:
            flush(force=True)

    flush(force=True)
    return chunks


def _tail_within_tokens(blocks: Sequence[DocumentBlock], max_tokens: int) -> List[DocumentBlock]:
    tail: List[DocumentBlock] = []
    total = 0
    for block in reversed(blocks):
        tokens = _estimate_tokens(block.text)
        if total + tokens > max_tokens and tail:
            break
        tail.append(block)
        total += tokens
    return list(reversed(tail))


def _build_section_path(blocks: Iterable[DocumentBlock]) -> str | None:
    headings = [block.text.strip() for block in blocks if block.kind == "heading"]
    if not headings:
        return None
    return " > ".join(dict.fromkeys(headings))


__all__ = ["ChunkingConfig", "chunk_document"]
