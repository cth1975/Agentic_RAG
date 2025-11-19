"""CLI entrypoints for ingestion (Phase 0) and retrieval (Phase 1)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .chunking import ChunkingConfig, chunk_document
from .config import PipelineSettings, load_settings
from .embeddings import MockEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from .indexing import AzureSearchIndexer, JSONLIndexer
from .models import DocumentRevision, EmbeddedChunk
from .retrieval import AzureSearchRetriever
from .reranking import EmbeddingSimilarityReranker, take_top_n
from .grounding import build_grounding_pack, summarize_grounding_pack

app = typer.Typer(help="Agentic RAG ingestion, indexing, and retrieval helpers")


def _resolve_settings(config_file: Optional[Path]) -> PipelineSettings:
    return load_settings(config_file)


def _build_embedding_provider(settings: PipelineSettings):
    if settings.mock_embeddings:
        return MockEmbeddingProvider()
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding.base_url,
        api_key=settings.embedding.api_key,
        model=settings.embedding.model,
    )


def _indexer_from_settings(settings: PipelineSettings, output_path: Optional[Path]):
    if output_path:
        return JSONLIndexer(output_path)
    if settings.azure_search:
        azure = settings.azure_search
        return AzureSearchIndexer(azure.endpoint, azure.api_key, azure.index_name)
    raise typer.BadParameter("Either --output-path or Azure Search settings are required")


def _close_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):  # pragma: no branch - trivial guard
        close()


@app.command()
def ingest_document(
    document_path: Path = typer.Argument(..., help="Path to normalized document JSON"),
    config_file: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Optional JSONL output. If omitted, Azure Search settings must be provided.",
    ),
    target_tokens: int = typer.Option(500, min=200, max=1200, help="Chunk target size"),
    overlap_tokens: int = typer.Option(75, min=0, max=200, help="Overlap tokens"),
) -> None:
    """Chunk, embed, and index a normalized document."""

    settings = _resolve_settings(config_file)
    chunking_config = ChunkingConfig(target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    document = DocumentRevision.from_json(document_path)
    chunks = chunk_document(document, chunking_config)
    typer.echo(f"Generated {len(chunks)} chunks for {document.doc_id} Rev {document.rev}")

    provider = _build_embedding_provider(settings)
    vectors = provider.embed([chunk.text for chunk in chunks])

    embedded_chunks: List[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            EmbeddedChunk(
                **chunk.__dict__,
                embedding=vector,
                embedding_model=settings.embedding.model,
            )
        )

    indexer = _indexer_from_settings(settings, output_path)
    indexer.upload(embedded_chunks)
    typer.echo(f"Uploaded {len(embedded_chunks)} chunks")
    _close_resource(indexer)
    _close_resource(provider)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="User question or search string."),
    config_file: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    user_group: List[str] = typer.Option(
        [],
        "--group",
        "-g",
        help="User groups used for ACL filtering (repeat for multiples).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        help="Override retrieval candidate pool (defaults to config.retrieval.candidate_k)",
    ),
    final_chunks: Optional[int] = typer.Option(
        None,
        help="Override number of context chunks kept after dedupe (defaults to config).",
    ),
    disable_rerank: bool = typer.Option(False, help="Bypass reranking stage."),
) -> None:
    """Run hybrid retrieval + reranking and preview a grounding pack."""

    settings = _resolve_settings(config_file)
    if not settings.azure_search:
        raise typer.BadParameter("Azure Search settings are required for querying")
    provider = _build_embedding_provider(settings)
    retriever = AzureSearchRetriever(
        settings.azure_search,
        provider,
        semantic_configuration=settings.retrieval.semantic_configuration,
    )
    candidate_k = top_k or settings.retrieval.candidate_k
    results = retriever.retrieve(query_text, user_group, top_k=candidate_k)
    typer.echo(f"Retrieved {len(results)} candidates from search")

    reranked = results
    rerank_chunks = final_chunks or settings.retrieval.final_context_chunks
    if settings.rerank.enabled and not disable_rerank:
        reranker = EmbeddingSimilarityReranker(provider)
        rerank_depth = min(settings.rerank.max_to_score, len(results))
        candidates = take_top_n(results, rerank_depth)
        reranked = reranker.rerank(query_text, candidates, rerank_depth)
        typer.echo(f"Reranked top {rerank_depth} candidates via embedding similarity")
    else:
        typer.echo("Reranking disabled; using search order")

    pack = build_grounding_pack(reranked, max_chunks=rerank_chunks)
    if not pack:
        typer.echo("No context chunks available after filtering")
    else:
        summaries = summarize_grounding_pack(pack)
        typer.echo(f"Prepared {len(pack)} context chunks:")
        for summary in summaries:
            typer.echo(summary)

    _close_resource(retriever)
    _close_resource(provider)


if __name__ == "__main__":
    app()
