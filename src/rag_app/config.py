"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from rag_app.exceptions import ConfigurationError


class DocumentConfig(BaseModel):
    input_dir: Path = Path("data/documents")
    glob_patterns: List[str] = Field(default_factory=lambda: ["**/*.pdf", "**/*.txt", "**/*.json"])
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_size(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class VectorStoreConfig(BaseModel):
    provider: Literal["faiss", "chroma"] = "faiss"
    path: Path = Path("storage/faiss_index")
    top_k: int = Field(default=5, gt=0)
    normalize_embeddings: bool = True
    allow_dangerous_deserialization: bool = True
    chroma_collection: str = "rag_documents"
    chroma_host: str = "api.trychroma.com"


class RetrievalConfig(BaseModel):
    mode: Literal["naive", "advanced", "corrective", "adaptive"] = "adaptive"
    advanced_fetch_multiplier: int = Field(default=4, ge=2, le=10)
    diversity_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    corrective_min_overlap: int = Field(default=1, ge=0)


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "local"] = "local"
    model: str = "text-embedding-3-small"
    local_dimensions: int = Field(default=384, ge=32, le=4096)


class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "extractive"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)


class LangSmithConfig(BaseModel):
    enabled: bool = True
    project_name: str = "naive-rag-production"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: Optional[Path] = Path("logs/rag.log")


class DomainConfig(BaseModel):
    name: str = "general"
    assistant_title: str = "Document RAG Assistant"
    min_source_documents: int = Field(default=1, ge=1)
    auto_index_on_startup: bool = True
    allow_uploads: bool = True
    system_prompt: str = (
        "You are a factual RAG assistant. Answer using only the retrieved context. "
        "If the context is insufficient, say so. Do not invent facts. "
        "Cite sources inline using [source:rank]."
    )


class AppConfig(BaseModel):
    documents: DocumentConfig = Field(default_factory=DocumentConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML or JSON configuration into a validated AppConfig."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config file does not exist: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(raw_text)
        else:
            payload = yaml.safe_load(raw_text) or {}
        config = AppConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - convert validation/parser errors to domain error.
        raise ConfigurationError(f"Failed to load config from {config_path}: {exc}") from exc

    if config.langsmith.enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", config.langsmith.project_name)
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return config
