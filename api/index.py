"""Vercel serverless RAG chatbot API.

This entrypoint is intentionally lightweight for Vercel Functions. It reads a
prebuilt document corpus from ``vercel_data/corpus.json``, performs lexical
retrieval, and uses Groq for grounded chat generation.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field


MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CORPUS_PATH = Path("vercel_data/corpus.json")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=5, gt=0, le=20)
    retrieval_mode: Literal["naive", "advanced", "corrective", "adaptive"] | None = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    top_k: int | None = Field(default=5, gt=0, le=20)
    retrieval_mode: Literal["naive", "advanced", "corrective", "adaptive"] | None = None


class IndexRequest(BaseModel):
    force: bool = True


app = FastAPI(title="Vercel RAG Chatbot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    corpus = _load_corpus()
    return {
        "documents": len(corpus.get("document_files", [])),
        "non_empty_documents": len(corpus.get("document_files", [])),
        "document_files": corpus.get("document_files", []),
        "index_exists": bool(corpus.get("chunks")),
        "vector_store_path": "vercel_data/corpus.json",
        "top_k": 5,
        "retrieval_mode": "adaptive",
        "embedding_provider": "lexical",
        "embedding_model": "serverless-keyword-retrieval",
        "llm_provider": "groq",
        "llm_model": MODEL,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", ""),
    }


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    corpus = _load_corpus()
    return {
        "documents": [
            {"name": Path(path).name, "path": path, "size_bytes": None, "modified_at": None}
            for path in corpus.get("document_files", [])
        ]
    }


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    del files
    raise HTTPException(
        status_code=400,
        detail=(
            "Uploads are disabled on the Vercel deployment because serverless "
            "filesystem changes do not persist. Add files locally, rebuild the corpus, and redeploy."
        ),
    )


@app.post("/api/index")
def index_documents(request: IndexRequest) -> dict[str, Any]:
    del request
    corpus = _load_corpus()
    return {
        "chunks": len(corpus.get("chunks", [])),
        "index_exists": bool(corpus.get("chunks")),
        "latency_ms": 0,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    return _answer(
        request.question,
        top_k=request.top_k or 5,
        chat_history="No prior conversation.",
        retrieval_mode=request.retrieval_mode or "adaptive",
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    user_messages = [message for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Chat needs at least one user message.")

    latest_question = user_messages[-1].content
    response = _answer(
        latest_question,
        top_k=request.top_k or 5,
        chat_history=_format_chat_history(request.messages),
        retrieval_mode=request.retrieval_mode or "adaptive",
    )
    response["message"] = {"role": "assistant", "content": response["answer"]}
    return response


def _answer(question: str, top_k: int, chat_history: str, retrieval_mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = _load_corpus()
    chunks = corpus.get("chunks", [])
    if not chunks:
        raise HTTPException(status_code=400, detail="No document corpus found. Rebuild and redeploy.")

    retrieval_started = time.perf_counter()
    routed_mode = None
    actual_mode = retrieval_mode
    if retrieval_mode == "adaptive":
        routed_mode = _classify_query(question)
        actual_mode = routed_mode

    documents, rewritten_query = _retrieve(question, chunks, top_k=top_k, mode=actual_mode)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    generation_started = time.perf_counter()
    answer, token_usage = _generate_answer(question, documents, chat_history)
    generation_ms = (time.perf_counter() - generation_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000

    return {
        "query": question,
        "answer": answer,
        "documents": documents,
        "token_usage": token_usage,
        "latency_ms": total_ms,
        "component_latency_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": total_ms,
        },
        "retrieval_mode": retrieval_mode,
        "rewritten_query": rewritten_query,
        "routed_mode": routed_mode,
    }


def _load_corpus() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        return {"document_files": [], "chunks": []}
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _retrieve(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    mode: str,
) -> tuple[list[dict[str, Any]], str | None]:
    if mode == "corrective":
        initial = _search(question, chunks, top_k=top_k, diversify=False)
        if _has_enough_overlap(question, initial):
            return _format_results(initial, mode), None

        rewritten_query = _rewrite_query(question)
        retry = _search(rewritten_query, chunks, top_k=top_k, diversify=True)
        return _format_results(retry, mode), rewritten_query

    diversify = mode in {"advanced", "corrective"}
    selected = _search(question, chunks, top_k=top_k, diversify=diversify)
    return _format_results(selected, mode), None


def _search(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    diversify: bool,
) -> list[tuple[float, dict[str, Any]]]:
    fetch_k = top_k * 4 if diversify else top_k
    query_terms = _terms(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_terms = list(chunk.get("terms") or _terms(chunk.get("content", "")))
        overlap = query_terms & set(chunk_terms)
        if not overlap:
            score = 0.0
        else:
            score = sum(1 + math.log(1 + chunk_terms.count(term)) for term in overlap)
            score = score / max(math.sqrt(len(set(chunk_terms))), 1)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if diversify:
        return _diversify(query_terms, scored[:fetch_k], top_k)
    return scored[:top_k]


def _format_results(
    selected: list[tuple[float, dict[str, Any]]],
    mode: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rank, (score, chunk) in enumerate(selected, start=1):
        results.append(
            {
                "content": chunk.get("content", ""),
                "metadata": {**chunk.get("metadata", {}), "retrieval_mode": mode},
                "score": float(score),
                "rank": rank,
            }
        )
    return results


def _has_enough_overlap(question: str, documents: list[dict[str, Any]]) -> bool:
    query_terms = _terms(question)
    if not query_terms:
        return bool(documents)
    best_overlap = max((len(query_terms & _terms(doc["content"])) for doc in documents), default=0)
    return best_overlap >= 1


def _classify_query(query: str) -> str:
    query_lower = query.lower()
    words = query_lower.split()
    if len(words) <= 2:
        return "naive"

    corrective_keywords = {
        "unrelated",
        "error",
        "bug",
        "fail",
        "wrong",
        "broken",
        "issue",
        "troubleshoot",
        "debug",
        "fix",
        "why does",
        "how to fix",
        "what is wrong",
        "corrupted",
    }
    if any(keyword in query_lower for keyword in corrective_keywords):
        return "corrective"

    advanced_keywords = {
        "compare",
        "contrast",
        "difference",
        "similar",
        "summarize",
        "summary",
        "list all",
        "all the",
        "explain the relationship",
        "pros and cons",
        "advantages",
        "disadvantages",
        "benefits",
        "features",
        "overview",
        "relationship between",
    }
    if any(keyword in query_lower for keyword in advanced_keywords):
        return "advanced"

    if len(words) > 12 or "," in query or " and " in query_lower or " or " in query_lower:
        return "advanced"

    return "naive"


def _diversify(
    query_terms: set[str],
    candidates: list[tuple[float, dict[str, Any]]],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    selected: list[tuple[float, dict[str, Any]]] = []
    remaining = candidates[:]
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")
        for index, (score, chunk) in enumerate(remaining):
            terms = _terms(chunk.get("content", ""))
            lexical_bonus = len(query_terms & terms) / max(len(query_terms), 1)
            diversity_penalty = max(
                (_jaccard(terms, _terms(selected_chunk.get("content", ""))) for _, selected_chunk in selected),
                default=0.0,
            )
            rerank_score = (0.7 * (score + lexical_bonus)) - (0.3 * diversity_penalty)
            if rerank_score > best_score:
                best_score = rerank_score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _rewrite_query(question: str) -> str:
    stopwords = {"about", "does", "from", "have", "please", "tell", "that", "the", "this", "what", "when", "where", "which", "with"}
    terms = sorted(term for term in _terms(question) if term not in stopwords)
    return " ".join(terms) or question


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _generate_answer(
    question: str,
    documents: list[dict[str, Any]],
    chat_history: str,
) -> tuple[str, dict[str, int | None]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_answer(documents), {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    context = "\n\n---\n\n".join(
        f"[source:{doc['rank']}] {doc['metadata'].get('source', 'unknown')}\n{doc['content']}"
        for doc in documents
    )
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual RAG chatbot. Answer using only the retrieved context. "
                    "If the context is insufficient, say so. Cite sources inline like [source:1]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{chat_history}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Retrieved context:\n{context}\n\nAnswer:"
                ),
            },
        ],
    )
    usage = completion.usage
    token_usage = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return completion.choices[0].message.content or "", token_usage


def _fallback_answer(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "The provided documents do not contain enough information."
    best = " ".join(documents[0].get("content", "").split())[:900]
    return f"Groq is not configured, but the top retrieved source says: {best} [source:1]"


def _format_chat_history(messages: list[ChatMessage], max_messages: int = 8) -> str:
    history = messages[:-1][-max_messages:]
    if not history:
        return "No prior conversation."
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def _terms(text: str) -> set[str]:
    return {term.lower() for term in TOKEN_RE.findall(text) if len(term) > 2}
