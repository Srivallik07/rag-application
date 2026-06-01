"""Grounded LLM answer generation."""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langsmith import traceable

from rag_app.exceptions import GenerationError
from rag_app.prompting import PromptBuilder
from rag_app.schemas import GenerationResult, RetrievedDocument, TokenUsage

logger = logging.getLogger(__name__)


class LLMChain:
    """Combine retrieved context with a user query and generate a grounded answer."""

    def __init__(
        self,
        llm: Runnable | None,
        prompt_builder: PromptBuilder,
        model_name: str,
        provider: str = "openai",
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.model_name = model_name
        self.provider = provider

    @traceable(run_type="llm", name="generate_grounded_answer")
    def generate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        chat_history: str | None = None,
    ) -> GenerationResult:
        """Generate a factual answer from retrieved documents."""
        messages = self.prompt_builder.build_messages(query, documents, chat_history=chat_history)
        prompt = self.prompt_builder.render_prompt(query, documents, chat_history=chat_history)
        started = perf_counter()

        if self.llm is None:
            answer = self._generate_extractive_answer(query, documents)
            latency_ms = (perf_counter() - started) * 1000
            return GenerationResult(
                answer=answer,
                prompt=prompt,
                model=self.model_name,
                token_usage=TokenUsage(),
                latency_ms=latency_ms,
            )

        try:
            llm_input: Any = messages if isinstance(self.llm, BaseChatModel) else prompt
            response = self.llm.invoke(
                llm_input,
                config={
                    "metadata": {
                        "ls_provider": self._provider_name(),
                        "ls_model_name": self.model_name,
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            if self._should_use_local_fallback(exc):
                logger.warning("LLM generation failed; using local extractive fallback: %s", exc)
                answer = self._generate_extractive_answer(query, documents)
                latency_ms = (perf_counter() - started) * 1000
                return GenerationResult(
                    answer=(
                        "OpenAI generation is unavailable for this account right now, so I used "
                        f"the local document fallback.\n\n{answer}"
                    ),
                    prompt=prompt,
                    model=f"{self.model_name}+extractive-fallback",
                    token_usage=TokenUsage(),
                    latency_ms=latency_ms,
                )
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        latency_ms = (perf_counter() - started) * 1000
        answer = str(getattr(response, "content", response))
        token_usage = self._extract_token_usage(response)

        logger.info("Generated answer in %.2f ms", latency_ms)
        logger.debug("LLM prompt: %s", prompt)
        logger.debug("LLM answer: %s", answer)

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            model=self.model_name,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    def _provider_name(self) -> str:
        if self.provider == "extractive":
            return "local"
        if self.llm is None:
            return "local"
        module = self.llm.__class__.__module__.lower()
        if "openai" in module:
            return "openai"
        if "groq" in module:
            return "groq"
        return self.llm.__class__.__name__

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return TokenUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        response_metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage") or {}
        return TokenUsage(
            input_tokens=token_usage.get("prompt_tokens"),
            output_tokens=token_usage.get("completion_tokens"),
            total_tokens=token_usage.get("total_tokens"),
        )

    @staticmethod
    def _should_use_local_fallback(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "insufficient_quota",
                "quota",
                "rate limit",
                "429",
            )
        )

    @staticmethod
    def _generate_extractive_answer(query: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return "The provided documents do not contain enough information to answer this question."

        query_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9]+", query.lower())
            if len(term) > 2
        }
        candidates: list[tuple[int, int, str]] = []
        for document in documents:
            sentences = re.split(r"(?<=[.!?])\s+", document.content.strip())
            for position, sentence in enumerate(sentences):
                cleaned = " ".join(sentence.split())
                if len(cleaned) < 30:
                    continue
                sentence_terms = set(re.findall(r"[a-zA-Z0-9]+", cleaned.lower()))
                score = len(query_terms & sentence_terms)
                candidates.append((score, document.rank * 1000 + position, f"{cleaned} [source:{document.rank}]"))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [candidate for candidate in candidates if candidate[0] > 0][:4]
        if not selected:
            top_document = " ".join(documents[0].content.split())[:700]
            return (
                "The retrieved context does not contain a direct match, but the most relevant "
                f"passage says: {top_document} [source:{documents[0].rank}]"
            )

        bullets = "\n".join(f"- {sentence}" for _, _, sentence in selected)
        return f"Based on the retrieved documents:\n{bullets}"
