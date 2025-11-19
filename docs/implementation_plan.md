# Enterprise Agentic RAG Implementation Plan

This document translates the requested Phase 0–4 roadmap into an actionable plan that keeps model choices swappable, including support for self-hosted OpenAI-compatible endpoints inside a private network.

## Design Principles
- **Model abstraction**: All LLM, embedding, reranker, and groundedness services are accessed through provider-neutral interfaces with configuration-driven endpoints and API keys. Adopt the OpenAI-compatible API surface (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank` if available) to make swapping between public OpenAI and private-deployed equivalents trivial.
- **Security-first retrieval**: Enforce document-level ACL filters in the search engine and propagate user groups on every query. Log retrieval IDs and user context for audit.
- **Traceability**: Stable document identities (`doc_id`, `rev`, `hash`, `source_url`) flow from ingestion through retrieval to citations.
- **Portability**: Core pipelines (OCR, chunking, embeddings, rerankers, graph, planner) are cloud-neutral. Azure AI Search is referenced for hybrid + ACL patterns, but the design also fits Elastic/OpenSearch with vector/BM25 hybrid search.

## Components and Model Swap Strategy
- **Embedding service**: Default to `text-embedding-3-large` or `BGE-M3` via an OpenAI-compatible embedding API. Endpoint, model name, and API key are read from configuration (e.g., `EMBEDDING_ENDPOINT`, `EMBEDDING_MODEL`).
- **Chat/completion LLM**: Use chat-completions through a configurable endpoint (`LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY`). Prompt caching and context compression are optional middlewares.
- **Reranker**: Support both hosted rerankers (e.g., Cohere) and self-hosted cross-encoders (e.g., `BAAI/bge-reranker-large`). Access via a small adapter interface: `score(query, documents) -> ranked_documents`. Configure with `RERANKER_ENDPOINT/MODEL`.
- **Groundedness/Content Safety**: Plug-in module with configurable endpoint (Azure Content Safety or equivalent). Toggle per-environment.
- **Planner (Agentic)**: Uses the same chat interface as the LLM but with its own system prompt and cache key; inherits endpoint/model from config to stay swappable.
- **GraphRAG extractors**: Relation/NER extraction via configurable LLM endpoint; community summary generation uses the same abstraction.

## Phase 0 — Data, Security, and Ingestion
- **Inventory** sources with metadata (owner, cadence, ACL provider, retention). Store in a catalog table.
- **Normalization pipeline**: Extract structured blocks (page, kind, text) from PDFs/DOCX/HTML; OCR scans with layout preservation (Azure Document Intelligence or Tesseract). Compute SHA-256 of raw text for change detection.
- **Document JSON** (per revision):
  ```json
  {
    "doc_id": "SPEC-1234",
    "rev": "D",
    "title": "Widget Spec",
    "effective_date": "2025-06-12",
    "allowed_groups": ["ME-Design", "QA-Compliance"],
    "source_url": "sharepoint://.../SPEC-1234 Rev D.pdf",
    "blocks": [{"page":1,"kind":"heading","text":"Scope"},{"page":1,"kind":"para","text":"..."}],
    "hash": "<sha256>"
  }
  ```
- **Chunking/Embeddings**: Heading-aware chunks (300–800 tokens, 10–20% overlap). Embed via configurable embedding endpoint. Persist `embedding_vector` alongside chunk metadata.
- **Indexing**: Hybrid search index fields include `doc_id`, `rev`, `chunk_id`, `page`, `section_path`, `text`, `embedding_vector`, `headings`, `table_markdown`, `effective_date`, `owner`, `source_url`, `hash`, `allowed_groups` (filterable), `tags`, timestamps. Enable BM25 + vector, RRF fusion, optional semantic ranker. Apply `allowed_groups` filter on every query.
- **Exit tests**: ACL trimming tests, OCR fidelity sampling, ingestion change-detection (only modified docs re-ingested).

## Phase 1 — Baseline RAG
- **Retrieval pipeline**: Hybrid recall (BM25 + vector) → RRF fusion → cross-encoder rerank → top-N (12–20) context pack. All calls use the configurable endpoints above.
- **Prompting**: System prompt enforces “answer only from provided sources; cite doc_id/rev/page.” Include a “cannot answer” branch.
- **Controls**: Context budget enforced; optional prompt compression (LLMLingua-2). Prompt caching for static system prompts.
- **Guardrails**: Pass user groups for ACL filters; allow-list tools; log retrieved chunk IDs and user context.
- **Exit tests**: Recall@50/MRR on gold set, SME citation validity, groundedness detector ≤2% ungrounded sentences, p95 latency ≤2.5s with caching.

## Phase 2 — Agentic RAG
- **Planner**: LLM-generated sub-queries with intents and filters; executed in parallel with the same hybrid+rERANK pipeline. Uses configurable LLM endpoint to remain swappable.
- **Router**: Simple heuristic (multi-facet/long queries → planner ON; simple queries → classic RAG).
- **Safety/Cost**: Max sub-queries, rerank depth caps, prompt caching for planner system prompt.
- **Exit tests**: Coverage ≥90% on multi-hop gold set, SME completeness uplift vs Phase 1, p95 latency within +800ms of Phase 1 at K=3 sub-queries.

## Phase 3 — GraphRAG
- **Graph construction**: Run NER + relation extraction over Phase-0 chunks; store triples in a graph DB with back-pointers to chunk/page anchors. Generate community summaries via the configurable LLM endpoint.
- **Query modes**: Global search (community summaries) and local search (entity neighborhoods), merged with hybrid+rERANK results for diversity.
- **Planner integration**: Router chooses graph-assisted mode for “connect-the-dots” asks; baseline for precise lookups.
- **Exit tests**: Path explainability, improved completeness on cross-doc synthesis, precision maintained.

## Phase 4 — Productionization
- **Evaluation harness**: Gold sets for grounded QA and synthesis; metrics include recall@K, MRR, citation validity, groundedness score, SME factuality. Use RAGAS/TruLens for logging and scoring.
- **Observability**: Trace sub-queries, retrieval hits, reranker scores, citations used, token usage, cache hits. Drift monitors tied to document revisions.
- **Guardrails**: ACL filters, groundedness filter pre-render, DLP/PII/export-control checks. Minimal agent toolset.
- **Cost/Latency**: Prompt caching for shared prompts, context compression when needed, tuned rerank window, optional response semantic cache.
- **Runbooks**: Re-index cadence with rollback plan; A/B for new chunkers/rerankers/embeddings; rollback to last good snapshot on ingest issues.

## Implementation Checklist (Swappable Models)
1. **Config schema**: Centralize model endpoints/names/keys in environment variables or a secrets store. Provide defaults for local dev (e.g., OpenAI base URL) and override for private endpoints.
2. **Client adapters**: Thin wrappers for chat, embeddings, rerank, and groundedness detectors implementing a stable interface. Each adapter supports OpenAI-compatible payloads and headers.
3. **Feature flags**: Toggle reranker, prompt compression, groundedness check, and planner per environment.
4. **Test harness**: Mocks for the adapter interfaces to validate pipelines without hitting real models; gold-set regression tests for retrieval and groundedness.
5. **Security plumbing**: Every retrieval call requires `allowed_groups`; audit log persists user, query, retrieved chunk IDs, and model versions used.
