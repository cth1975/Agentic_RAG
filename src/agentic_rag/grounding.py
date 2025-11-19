"""Helpers to assemble grounding packs from search results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .retrieval import SearchResult


@dataclass(slots=True)
class GroundingChunk:
    """Metadata sent to the LLM along with the snippet text."""

    chunk_id: str
    doc_id: str
    rev: str
    text: str
    section_path: str | None
    pages: Tuple[int | None, int | None]
    source_url: str
    citation: str
    score: float


def build_grounding_pack(
    candidates: Sequence[SearchResult],
    *,
    max_chunks: int,
    diversify: bool = True,
) -> List[GroundingChunk]:
    """Deduplicate chunks and enforce light-weight source diversity."""

    seen: set[tuple[str, str | None, int | None, int | None]] = set()
    per_doc_counts: dict[str, int] = {}
    pack: List[GroundingChunk] = []
    for candidate in candidates:
        key = (candidate.doc_id, candidate.section_path, candidate.page_start, candidate.page_end)
        if key in seen:
            continue
        seen.add(key)
        if diversify:
            count = per_doc_counts.get(candidate.doc_id, 0)
            if count >= max(2, max_chunks // 3):
                continue
            per_doc_counts[candidate.doc_id] = count + 1
        citation = _format_citation(candidate)
        pages = (candidate.page_start, candidate.page_end)
        pack.append(
            GroundingChunk(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                rev=candidate.rev,
                text=candidate.text,
                section_path=candidate.section_path,
                pages=pages,
                source_url=candidate.source_url,
                citation=citation,
                score=candidate.score,
            )
        )
        if len(pack) >= max_chunks:
            break
    return pack


def summarize_grounding_pack(chunks: Iterable[GroundingChunk]) -> List[str]:
    """Generate printable summaries for CLI output."""

    summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        page_start, page_end = chunk.pages
        if page_start and page_end and page_start != page_end:
            page_label = f"pp.{page_start}-{page_end}"
        elif page_start:
            page_label = f"p.{page_start}"
        else:
            page_label = "pages n/a"
        snippet = " ".join(chunk.text.strip().split())[:260]
        summaries.append(
            f"[{idx}] {chunk.citation} ({page_label}) score={chunk.score:.3f}\n    {snippet}"
        )
    return summaries


def _format_citation(candidate: SearchResult) -> str:
    rev_part = f" Rev {candidate.rev}" if candidate.rev else ""
    page = f" p.{candidate.page_start}" if candidate.page_start else ""
    return f"{candidate.doc_id}{rev_part}{page}"


__all__ = ["GroundingChunk", "build_grounding_pack", "summarize_grounding_pack"]
