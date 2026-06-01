"""Embedding and LLM provider factories."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_app.config import AppConfig
from rag_app.exceptions import ConfigurationError


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for offline indexing and retrieval."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
            if len(token) <= 2:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_embeddings(config: AppConfig) -> Embeddings:
    """Build an embeddings provider from configuration."""
    if config.embeddings.provider == "local":
        return LocalHashEmbeddings(dimensions=config.embeddings.local_dimensions)
    if config.embeddings.provider == "openai":
        return OpenAIEmbeddings(
            model=config.embeddings.model,
        )
    raise ConfigurationError(f"Unsupported embeddings provider: {config.embeddings.provider}")


def build_llm(config: AppConfig) -> Runnable:
    """Build a chat model from configuration."""
    if config.llm.provider == "openai":
        return ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout_seconds,
        )
    if config.llm.provider == "groq":
        return ChatGroq(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported LLM provider: {config.llm.provider}")
