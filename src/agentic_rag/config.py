"""Lightweight configuration loader without external dependencies."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional dependency used only when a YAML file is provided
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ENV_PREFIX = "AGENTIC_RAG_"


@dataclass(slots=True)
class ModelEndpointSettings:
    base_url: str
    api_key: str
    model: str
    deployment: Optional[str] = None


@dataclass(slots=True)
class AzureSearchSettings:
    endpoint: str
    api_key: str
    index_name: str


@dataclass(slots=True)
class RetrievalSettings:
    """Tunable knobs for the hybrid retrieval pipeline."""

    candidate_k: int = 60
    final_context_chunks: int = 12
    semantic_configuration: Optional[str] = None


@dataclass(slots=True)
class RerankSettings:
    """Controls how many candidates feed into the reranker stage."""

    enabled: bool = True
    max_to_score: int = 40


@dataclass(slots=True)
class PipelineSettings:
    embedding: ModelEndpointSettings
    environment: str = "dev"
    mock_embeddings: bool = False
    azure_search: Optional[AzureSearchSettings] = None
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)


def _load_yaml(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not yaml:
        raise RuntimeError(
            "PyYAML is not installed. Install it or avoid passing --config-file."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _load_embedding(data: Dict[str, Any]) -> ModelEndpointSettings:
    cfg = data.get("embedding", {})
    return ModelEndpointSettings(
        base_url=_env_or("EMBEDDING__BASE_URL", cfg.get("base_url")),
        api_key=_env_or("EMBEDDING__API_KEY", cfg.get("api_key")),
        model=_env_or("EMBEDDING__MODEL", cfg.get("model")),
        deployment=_env_or("EMBEDDING__DEPLOYMENT", cfg.get("deployment")),
    )


def _load_azure_search(data: Dict[str, Any]) -> Optional[AzureSearchSettings]:
    cfg = data.get("azure_search") or {}
    endpoint = _env_or("AZURE_SEARCH__ENDPOINT", cfg.get("endpoint"))
    api_key = _env_or("AZURE_SEARCH__API_KEY", cfg.get("api_key"))
    index_name = _env_or("AZURE_SEARCH__INDEX_NAME", cfg.get("index_name"))
    if not all([endpoint, api_key, index_name]):
        return None
    return AzureSearchSettings(endpoint=endpoint, api_key=api_key, index_name=index_name)


def _load_retrieval(data: Dict[str, Any]) -> RetrievalSettings:
    cfg = data.get("retrieval") or {}
    candidate_k = int(cfg.get("candidate_k", 60))
    final_context_chunks = int(cfg.get("final_context_chunks", 12))
    semantic_configuration = cfg.get("semantic_configuration")
    env_semantic = os.getenv(f"{ENV_PREFIX}RETRIEVAL__SEMANTIC_CONFIGURATION")
    if env_semantic:
        semantic_configuration = env_semantic
    env_candidate = os.getenv(f"{ENV_PREFIX}RETRIEVAL__CANDIDATE_K")
    if env_candidate:
        candidate_k = int(env_candidate)
    env_final = os.getenv(f"{ENV_PREFIX}RETRIEVAL__FINAL_CONTEXT_CHUNKS")
    if env_final:
        final_context_chunks = int(env_final)
    return RetrievalSettings(
        candidate_k=candidate_k,
        final_context_chunks=final_context_chunks,
        semantic_configuration=semantic_configuration,
    )


def _load_rerank(data: Dict[str, Any]) -> RerankSettings:
    cfg = data.get("rerank") or {}
    enabled = _read_bool(
        os.getenv(f"{ENV_PREFIX}RERANK__ENABLED"),
        cfg.get("enabled", True),
    )
    max_to_score = int(
        os.getenv(f"{ENV_PREFIX}RERANK__MAX_TO_SCORE")
        or cfg.get("max_to_score", 40)
    )
    return RerankSettings(enabled=enabled, max_to_score=max_to_score)


def _env_or(key: str, default: Optional[str]) -> str:
    value = os.getenv(f"{ENV_PREFIX}{key}")
    if value is not None:
        return value
    if default is None:
        raise RuntimeError(f"Missing required config value for {key}")
    return default


def load_settings(config_path: Optional[Path] = None) -> PipelineSettings:
    data = _load_yaml(config_path)
    embedding = _load_embedding(data)
    environment = os.getenv(f"{ENV_PREFIX}ENVIRONMENT", data.get("environment", "dev")).lower()
    mock_embeddings = _read_bool(
        os.getenv(f"{ENV_PREFIX}MOCK_EMBEDDINGS"), data.get("mock_embeddings", False)
    )
    azure_search = _load_azure_search(data)
    retrieval = _load_retrieval(data)
    rerank = _load_rerank(data)
    return PipelineSettings(
        embedding=embedding,
        environment=environment,
        mock_embeddings=mock_embeddings,
        azure_search=azure_search,
        retrieval=retrieval,
        rerank=rerank,
    )


__all__ = [
    "ModelEndpointSettings",
    "AzureSearchSettings",
    "PipelineSettings",
    "RetrievalSettings",
    "RerankSettings",
    "load_settings",
]
