"""Semantic retrieval over FAISS."""

from __future__ import annotations

import logging
import re
from time import perf_counter

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable
from langsmith import traceable

from rag_app.config import RetrievalConfig, VectorStoreConfig
from rag_app.exceptions import VectorStoreError
from rag_app.schemas import RetrievalResult, RetrievedDocument

logger = logging.getLogger(__name__)


class Retriever:
    """Run top-k semantic search and return ranked documents with metadata."""

    def __init__(
        self,
        vector_store: FAISS,
        config: VectorStoreConfig,
        retrieval_config: RetrievalConfig | None = None,
        llm: Runnable | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.config = config
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.llm = llm

    @traceable(run_type="retriever", name="semantic_search")
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        pool_size: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k relevant documents for a query."""
        cleaned_query = self._process_query(query)
        k = top_k or self.config.top_k
        selected_mode = mode or self.retrieval_config.mode
        started = perf_counter()

        try:
            routed_mode = None
            if selected_mode == "adaptive":
                routed_mode = self.classify_query(cleaned_query)
                logger.info("Adaptive RAG classified query complexity; routing query %r to %s", cleaned_query, routed_mode)
                actual_mode = routed_mode
            else:
                actual_mode = selected_mode

            if actual_mode == "advanced":
                documents = self._advanced_retrieve(cleaned_query, k, pool_size=pool_size)
                rewritten_query = None
            elif actual_mode == "corrective":
                documents, rewritten_query = self._corrective_retrieve(
                    cleaned_query,
                    k,
                    pool_size=pool_size,
                )
            else:
                fetch_k = pool_size or k
                documents = self._similarity_retrieve(cleaned_query, fetch_k, mode="naive")[:k]
                rewritten_query = None

            if min_score is not None:
                documents = self._filter_by_min_score(documents, min_score)[:k]
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "Retrieved %d documents for query=%r with mode=%s%s in %.2f ms",
            len(documents),
            cleaned_query,
            selected_mode,
            f" (routed={routed_mode})" if routed_mode else "",
            latency_ms,
        )
        logger.debug("Retrieval results: %s", [doc.model_dump() for doc in documents])
        return RetrievalResult(
            query=cleaned_query,
            documents=documents,
            latency_ms=latency_ms,
            mode=selected_mode,
            rewritten_query=rewritten_query,
            routed_mode=routed_mode,
        )

    def classify_query(self, query: str) -> str:
        """Classify query complexity into 'naive', 'advanced', or 'corrective'."""
        if self.llm is None:
            return self._heuristic_classify(query)

        prompt = (
            "You are a query router in a Retrieval-Augmented Generation (RAG) system.\n"
            "Your task is to classify the input query into one of three retrieval strategies:\n"
            "- 'naive': Simple, direct factual questions that can be answered with a direct search (e.g., 'What is the refund window?', 'Who is the CEO?').\n"
            "- 'advanced': Questions that require comparing, contrasting, summarizing, or synthesizing multiple pieces of information (e.g., 'Compare the refund policy with the exchange rules.', 'What are the main benefits?').\n"
            "- 'corrective': Queries that are complex, ambiguous, or likely out-of-domain, requiring rewriting or keyword-based fallback (e.g., 'what is completely unrelated and complex query?').\n\n"
            f"Query: {query}\n\n"
            "Respond with exactly one word from ('naive', 'advanced', 'corrective') and nothing else. Do not include formatting, punctuation, or explanations."
        )
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = getattr(response, "content", str(response)).strip().lower()
            response_text = re.sub(r"[^\w\s-]", "", response_text).strip()
            if response_text in {"naive", "advanced", "corrective"}:
                logger.info("LLM classified query complexity as: %s", response_text)
                return response_text
            logger.warning("LLM returned invalid classification: %s. Falling back to heuristics.", response_text)
        except Exception as exc:
            logger.warning("LLM query classification failed: %s. Falling back to heuristics.", exc)

        return self._heuristic_classify(query)

    def _heuristic_classify(self, query: str) -> str:
        """Fallback rule-based classification based on keywords and query structure."""
        query_lower = query.lower()
        words = query_lower.split()

        # If query is extremely short (e.g. 1-2 words), route to naive
        if len(words) <= 2:
            return "naive"

        # Corrective keywords (ambiguous, complex, out-of-domain, or specific error-prone terms)
        corrective_keywords = {
            "unrelated", "error", "bug", "fail", "wrong", "broken", "issue", "troubleshoot",
            "debug", "fix", "why does", "how to fix", "what is wrong", "corrupted"
        }
        if any(keyword in query_lower for keyword in corrective_keywords):
            return "corrective"

        # Advanced keywords (requiring synthesis, comparison, listing all)
        advanced_keywords = {
            "compare", "contrast", "difference", "similar", "summarize", "summary",
            "list all", "all the", "explain the relationship", "pros and cons", "advantages",
            "disadvantages", "benefits", "features", "overview", "relationship between"
        }
        if any(keyword in query_lower for keyword in advanced_keywords):
            return "advanced"

        # If it contains lots of conjunctions or punctuation, might be complex -> advanced/corrective
        if len(words) > 12 or "," in query or " and " in query_lower or " or " in query_lower:
            return "advanced"

        return "naive"

    def _similarity_retrieve(self, query: str, k: int, mode: str) -> list[RetrievedDocument]:
        scored_docs = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata={**doc.metadata, "retrieval_mode": mode},
                score=float(score),
                rank=index + 1,
            )
            for index, (doc, score) in enumerate(scored_docs)
        ]

    def _advanced_retrieve(
        self,
        query: str,
        k: int,
        pool_size: int | None = None,
    ) -> list[RetrievedDocument]:
        fetch_k = pool_size or max(k, k * self.retrieval_config.advanced_fetch_multiplier)
        candidates = self._similarity_retrieve(query, fetch_k, mode="advanced")
        query_terms = self._terms(query)
        ranked = self._diversify(query_terms, candidates, k)
        return self._rerank(ranked, mode="advanced")

    def _corrective_retrieve(
        self,
        query: str,
        k: int,
        pool_size: int | None = None,
    ) -> tuple[list[RetrievedDocument], str | None]:
        initial = self._similarity_retrieve(query, k, mode="corrective")
        if self._has_enough_overlap(query, initial):
            return self._annotate(initial, status="accepted"), None

        rewritten_query = self._rewrite_query(query)
        fetch_k = pool_size or max(k, k * self.retrieval_config.advanced_fetch_multiplier)
        retry = self._similarity_retrieve(rewritten_query, fetch_k, mode="corrective")
        diversified = self._diversify(self._terms(rewritten_query), retry, k)
        return self._annotate(
            self._rerank(diversified, mode="corrective"),
            status="retried",
            rewritten_query=rewritten_query,
        ), rewritten_query

    def _diversify(
        self,
        query_terms: set[str],
        candidates: list[RetrievedDocument],
        k: int,
    ) -> list[RetrievedDocument]:
        selected: list[RetrievedDocument] = []
        remaining = candidates[:]
        while remaining and len(selected) < k:
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                relevance = self._relevance_score(query_terms, candidate)
                diversity_penalty = max(
                    (self._jaccard(self._terms(candidate.content), self._terms(doc.content)) for doc in selected),
                    default=0.0,
                )
                score = (
                    self.retrieval_config.diversity_lambda * relevance
                    - (1 - self.retrieval_config.diversity_lambda) * diversity_penalty
                )
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def _has_enough_overlap(self, query: str, documents: list[RetrievedDocument]) -> bool:
        query_terms = self._terms(query)
        if not query_terms:
            return bool(documents)
        best_overlap = max((len(query_terms & self._terms(doc.content)) for doc in documents), default=0)
        return best_overlap >= self.retrieval_config.corrective_min_overlap

    @staticmethod
    def _rewrite_query(query: str) -> str:
        stopwords = {
            "about",
            "does",
            "from",
            "have",
            "how",
            "please",
            "tell",
            "that",
            "the",
            "their",
            "there",
            "this",
            "what",
            "when",
            "where",
            "which",
            "with",
            "would",
        }
        terms = sorted(term for term in Retriever._terms(query) if term not in stopwords)
        return " ".join(terms) or query

    @staticmethod
    def _relevance_score(query_terms: set[str], document: RetrievedDocument) -> float:
        vector_score = 1.0 / (1.0 + max(document.score, 0.0))
        lexical_bonus = len(query_terms & Retriever._terms(document.content)) / max(len(query_terms), 1)
        return vector_score + lexical_bonus

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _rerank(documents: list[RetrievedDocument], mode: str) -> list[RetrievedDocument]:
        return [
            document.model_copy(
                update={
                    "rank": index + 1,
                    "metadata": {**document.metadata, "retrieval_mode": mode},
                }
            )
            for index, document in enumerate(documents)
        ]

    @staticmethod
    def _annotate(
        documents: list[RetrievedDocument],
        status: str,
        rewritten_query: str | None = None,
    ) -> list[RetrievedDocument]:
        extra = {"retrieval_status": status}
        if rewritten_query:
            extra["rewritten_query"] = rewritten_query
        return [
            document.model_copy(update={"metadata": {**document.metadata, **extra}})
            for document in documents
        ]

    @staticmethod
    def _filter_by_min_score(
        documents: list[RetrievedDocument],
        min_score: float,
    ) -> list[RetrievedDocument]:
        filtered = [
            document
            for document in documents
            if Retriever._relevance_score(set(), document) >= min_score
            or (1.0 / (1.0 + max(document.score, 0.0))) >= min_score
        ]
        return filtered or documents

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term.lower() for term in re.findall(r"[a-zA-Z0-9_]+", text) if len(term) > 2}

    @staticmethod
    def _process_query(query: str) -> str:
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned
