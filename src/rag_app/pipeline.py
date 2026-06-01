"""RAG pipeline orchestrator."""

from __future__ import annotations

import logging
from time import perf_counter

from langsmith import traceable

from rag_app.config import AppConfig
from rag_app.document_loader import DocumentLoader
from rag_app.llm_chain import LLMChain
from rag_app.prompting import PromptBuilder
from rag_app.providers import build_embeddings, build_llm
from rag_app.retriever import Retriever
from rag_app.schemas import RAGResponse
from rag_app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level orchestrator for indexing, retrieval, and generation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.document_loader = DocumentLoader(config.documents)
        self.embeddings = build_embeddings(config)
        self.vector_store_manager = VectorStoreManager(config.vector_store, self.embeddings)
        self.prompt_builder = PromptBuilder(system_prompt=config.domain.system_prompt)
        self._llm_chain: LLMChain | None = None

    @property
    def llm_chain(self) -> LLMChain:
        """Build the LLM chain only when generation is requested."""
        if self._llm_chain is None:
            llm = None if self.config.llm.provider == "extractive" else build_llm(self.config)
            self._llm_chain = LLMChain(
                llm,
                self.prompt_builder,
                self.config.llm.model,
                provider=self.config.llm.provider,
            )
        return self._llm_chain

    @traceable(run_type="chain", name="rag_index_documents")
    def index_documents(self, force_rebuild: bool = False) -> int:
        """Load local files, split them, embed chunks, and store them in FAISS."""
        documents = self.document_loader.load()
        if force_rebuild or not self.vector_store_manager.index_exists():
            self.vector_store_manager.build(documents)
        else:
            self.vector_store_manager.load()
            logger.info("Existing FAISS index found; use force_rebuild=True to rebuild.")
        return len(documents)

    @traceable(run_type="chain", name="rag_answer_question")
    def answer(
        self,
        query: str,
        top_k: int | None = None,
        chat_history: str | None = None,
        retrieval_mode: str | None = None,
        pool_size: int | None = None,
        min_score: float | None = None,
    ) -> RAGResponse:
        """Run the complete RAG flow for a single user query."""
        total_started = perf_counter()
        if self.vector_store_manager._store is None:
            self.vector_store_manager.load()

        retriever = Retriever(
            self.vector_store_manager.store,
            self.config.vector_store,
            self.config.retrieval,
            llm=self.llm_chain.llm,
        )
        retrieval = retriever.retrieve(
            query,
            top_k=top_k,
            mode=retrieval_mode,
            pool_size=pool_size,
            min_score=min_score,
        )
        generation = self.llm_chain.generate(
            retrieval.query,
            retrieval.documents,
            chat_history=chat_history,
        )
        total_latency_ms = (perf_counter() - total_started) * 1000

        return RAGResponse(
            query=retrieval.query,
            answer=generation.answer,
            documents=retrieval.documents,
            token_usage=generation.token_usage,
            latency_ms=total_latency_ms,
            component_latency_ms={
                "retrieval": retrieval.latency_ms,
                "generation": generation.latency_ms,
                "total": total_latency_ms,
            },
            retrieval_mode=retrieval.mode,
            rewritten_query=retrieval.rewritten_query,
            routed_mode=retrieval.routed_mode,
        )
