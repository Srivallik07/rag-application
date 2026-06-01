"""Pydantic schemas returned by the RAG pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int


class RetrievalResult(BaseModel):
    query: str
    documents: List[RetrievedDocument]
    latency_ms: float
    mode: str = "naive"
    rewritten_query: Optional[str] = None
    routed_mode: Optional[str] = None


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class GenerationResult(BaseModel):
    answer: str
    prompt: str
    model: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    documents: List[RetrievedDocument]
    token_usage: TokenUsage
    latency_ms: float
    component_latency_ms: Dict[str, float]
    retrieval_mode: str = "naive"
    rewritten_query: Optional[str] = None
    routed_mode: Optional[str] = None
