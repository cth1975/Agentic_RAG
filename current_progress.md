# Current Progress

## Retrieval bootstrap (2025-11-19)
- Extended `PipelineSettings` so reranker and retrieval knobs live in config (YAML + env overrides).
- Added Azure Search retriever, embedding-similarity reranker, and grounding-pack helpers wired into the Typer CLI.
- `agentic-rag query` now runs hybrid retrieval + reranking to preview security-trimmed grounding packs, advancing Phase 1 of the plan.

## Next focus areas
- Wire evaluation harness + smoke tests for the retrieval CLI once live data is connected.
- Extend the grounding pack into a chat-ready prompt + groundedness checker to satisfy Phase 1 exit criteria.
