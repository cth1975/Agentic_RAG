# Agentic RAG

This repository hosts an enterprise-ready Agentic RAG implementation roadmap and
reference scaffolding so platform teams can iterate toward the Phase 0–4 plan
outlined in `docs/implementation_plan.md`.

## What was added in this iteration?

* `docs/implementation_plan.md` – multi-phase blueprint covering ingestion,
  retrieval, agentic planning, GraphRAG, and productionization.
* `src/agentic_rag` – Python package that begins implementing Phase 0–1 by
  modeling normalized documents, chunking them, generating embeddings via any
  OpenAI-compatible endpoint (or a deterministic mock for CI), shipping chunks
  into Azure AI Search or local JSONL files, and now querying Azure Search with
  hybrid retrieval + reranking to preview grounding packs.
* `pyproject.toml` – exposes a CLI `agentic-rag ingest-document` entrypoint to
  operationalize the ingestion flow.

## Getting started

1. Create a virtual environment and install dependencies: `pip install -e .`
2. Provide settings via environment variables or a YAML file (see below).
3. Normalize a document into the JSON shape described in Phase 0.2.
4. Run the CLI to chunk, embed, and export/index the document.
5. Use the `query` sub-command to exercise the hybrid retrieval + reranking
   pipeline and preview grounding packs.

```bash
agentic-rag ingest-document normalized_doc.json \
  --config-file config/settings.yaml \
  --output-path artifacts/spec_chunks.jsonl \
  --target-tokens 600 --overlap-tokens 90
```

Preview retrieval + reranking results:

```bash
agentic-rag query "What torque changes shipped in Rev D?" \
  --config-file config/settings.yaml \
  --group ME-Design --group QA-Compliance \
  --top-k 80 --final-chunks 15
```

## Configuration

All settings are controlled via the `PipelineSettings` dataclass in
`agentic_rag/config.py`. You can provide values in a YAML file passed to the CLI
via `--config-file` *and/or* via environment variables prefixed with
`AGENTIC_RAG_`.

Example YAML (`config/settings.yaml`):

```yaml
environment: dev
mock_embeddings: true
embedding:
  base_url: http://localhost:11434
  api_key: dev-token
  model: text-embedding-3-large
azure_search:
  endpoint: https://my-search.search.windows.net
  api_key: $SEARCH_KEY
  index_name: engineering-specs
retrieval:
  candidate_k: 60
  final_context_chunks: 12
  semantic_configuration: default
rerank:
  enabled: true
  max_to_score: 40
```

Setting `mock_embeddings: true` ensures air-gapped or CI runs produce stable
vectors without calling the managed service. When deploying to staging or prod,
flip the flag off and point `embedding.base_url` to your OpenAI-compatible
endpoint (Azure OpenAI, on-prem gateway, etc.).

## Next steps

With the ingestion skeleton in place we can implement:

* Incremental ingestion monitors (Phase 0.4).
* Retrieval + reranking service with ACL-aware security trimming (Phase 1).
* Planner/orchestrator for agentic retrieval (Phase 2).
* Graph construction and dual-mode (global/local) retrieval (Phase 3).
* Evaluation harness, guardrails, and ops tooling (Phase 4).
