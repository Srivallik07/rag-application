"""FastAPI backend for the RAG application."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Any, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_app.config import AppConfig, load_config
from rag_app.exceptions import RAGError
from rag_app.logging_config import configure_logging
from rag_app.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=20)
    retrieval_mode: Literal["naive", "advanced", "corrective", "adaptive"] | None = None
    pool_size: int | None = Field(default=None, gt=0, le=200)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=20)
    retrieval_mode: Literal["naive", "advanced", "corrective", "adaptive"] | None = None
    pool_size: int | None = Field(default=None, gt=0, le=200)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class IndexRequest(BaseModel):
    force: bool = True


class UploadResponse(BaseModel):
    saved_files: list[str]


class AppState:
    """Lazy application state so import does not require API credentials."""

    def __init__(self) -> None:
        self.config: AppConfig | None = None
        self.pipeline: RAGPipeline | None = None

    def get_config(self) -> AppConfig:
        if self.config is None:
            load_dotenv(dotenv_path=".env")
            self.config = load_config("config.yaml")
            configure_logging(self.config.logging)
        return self.config

    def get_pipeline(self) -> RAGPipeline:
        if self.pipeline is None:
            self.pipeline = RAGPipeline(self.get_config())
        return self.pipeline

    def reset_pipeline(self) -> None:
        self.pipeline = None


state = AppState()


def _ensure_domain_index() -> None:
    """Build or load the backend corpus index on API startup."""
    config = state.get_config()
    document_files = [path for path in _document_files(config) if path.stat().st_size > 0]
    if len(document_files) < config.domain.min_source_documents:
        logger.warning(
            "Found %d documents but expected at least %d for domain=%s",
            len(document_files),
            config.domain.min_source_documents,
            config.domain.name,
        )
        return
    if not config.domain.auto_index_on_startup:
        return

    pipeline = state.get_pipeline()
    if not pipeline.vector_store_manager.index_exists():
        logger.info("Indexing %d %s documents on startup...", len(document_files), config.domain.name)
        pipeline.index_documents(force_rebuild=True)
        return

    if pipeline.vector_store_manager._store is None:
        try:
            pipeline.vector_store_manager.load()
        except Exception:
            logger.exception("Failed to load index; rebuilding.")
            pipeline.index_documents(force_rebuild=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(dotenv_path=".env")
    _ensure_domain_index()
    yield


app = FastAPI(title="Healthcare RAG API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    config = state.get_config()
    pipeline = state.get_pipeline()
    documents = _document_files(config)
    non_empty_documents = [path for path in documents if path.stat().st_size > 0]
    return {
        "documents": len(documents),
        "non_empty_documents": len(non_empty_documents),
        "document_files": [_relative_to_workspace(path) for path in documents],
        "index_exists": pipeline.vector_store_manager.index_exists(),
        "vector_store_provider": config.vector_store.provider,
        "vector_store_path": (
            f"chroma://{config.vector_store.chroma_host}/"
            f"{os.getenv('CHROMA_DATABASE', 'unknown')}/{config.vector_store.chroma_collection}"
            if config.vector_store.provider == "chroma"
            else str(config.vector_store.path)
        ),
        "top_k": config.vector_store.top_k,
        "retrieval_mode": config.retrieval.mode,
        "embedding_provider": config.embeddings.provider,
        "embedding_model": config.embeddings.model,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", config.langsmith.project_name),
        "domain": config.domain.name,
        "assistant_title": config.domain.assistant_title,
        "allow_uploads": config.domain.allow_uploads,
        "min_source_documents": config.domain.min_source_documents,
        "corpus_ready": len(non_empty_documents) >= config.domain.min_source_documents,
    }


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    config = state.get_config()
    files = _document_files(config)
    return {
        "documents": [
            {
                "name": path.name,
                "path": _relative_to_workspace(path),
                "size_bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            }
            for path in files
        ]
    }


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    config = state.get_config()
    if not config.domain.allow_uploads:
        raise HTTPException(
            status_code=403,
            detail=(
                "Uploads are disabled. This deployment uses a fixed backend healthcare corpus "
                f"with {config.domain.min_source_documents}+ source documents."
            ),
        )
    config.documents.input_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        filename = _safe_filename(upload.filename or "document")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".json"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {filename}. Use PDF, TXT, or JSON.",
            )

        destination = _unique_path(config.documents.input_dir / filename)
        content = await upload.read()
        destination.write_bytes(content)
        saved.append(_relative_to_workspace(destination))

    state.reset_pipeline()
    return UploadResponse(saved_files=saved)


@app.post("/api/index")
def index_documents(request: IndexRequest) -> dict[str, Any]:
    config = state.get_config()
    document_files = _document_files(config)
    if not document_files:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Upload PDF, TXT, or JSON files first.",
        )
    if not any(path.stat().st_size > 0 for path in document_files):
        raise HTTPException(
            status_code=400,
            detail="Documents were found, but they are empty. Add text to a file or upload a non-empty document.",
        )

    started = time()
    try:
        pipeline = state.get_pipeline()
        chunk_count = pipeline.index_documents(force_rebuild=request.force)
    except RAGError as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected indexing failure")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return {
        "chunks": chunk_count,
        "index_exists": pipeline.vector_store_manager.index_exists(),
        "latency_ms": round((time() - started) * 1000, 2),
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    pipeline = state.get_pipeline()
    if not pipeline.vector_store_manager.index_exists():
        raise HTTPException(status_code=400, detail="No vector index found. Run indexing first.")

    try:
        response = pipeline.answer(
            request.question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            pool_size=request.pool_size,
            min_score=request.min_score,
        )
    except RAGError as exc:
        logger.exception("RAG answer failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected answer failure")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc

    return response.model_dump()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    user_messages = [message for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Chat needs at least one user message.")

    latest_question = user_messages[-1].content
    response = _answer_with_history(
        latest_question,
        top_k=request.top_k,
        chat_history=_format_chat_history(request.messages),
        retrieval_mode=request.retrieval_mode,
        pool_size=request.pool_size,
        min_score=request.min_score,
    )
    return {
        "message": {
            "role": "assistant",
            "content": response["answer"],
        },
        "answer": response["answer"],
        "documents": response["documents"],
        "token_usage": response["token_usage"],
        "latency_ms": response["latency_ms"],
        "component_latency_ms": response["component_latency_ms"],
        "retrieval_mode": response["retrieval_mode"],
        "rewritten_query": response["rewritten_query"],
        "routed_mode": response.get("routed_mode"),
    }


def _answer_with_history(
    question: str,
    top_k: int | None,
    chat_history: str | None,
    retrieval_mode: str | None,
    pool_size: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    pipeline = state.get_pipeline()
    if not pipeline.vector_store_manager.index_exists():
        raise HTTPException(status_code=400, detail="No vector index found. Run indexing first.")

    try:
        response = pipeline.answer(
            question,
            top_k=top_k,
            chat_history=chat_history,
            retrieval_mode=retrieval_mode,
            pool_size=pool_size,
            min_score=min_score,
        )
    except RAGError as exc:
        logger.exception("RAG chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected chat failure")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}") from exc

    return response.model_dump()


def _format_chat_history(messages: list[ChatMessage], max_messages: int = 8) -> str:
    history = messages[:-1][-max_messages:]
    if not history:
        return "No prior conversation."
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def _document_files(config: AppConfig) -> list[Path]:
    files: list[Path] = []
    for pattern in config.documents.glob_patterns:
        files.extend(path for path in config.documents.input_dir.glob(pattern) if path.is_file())
    return sorted(set(files))


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "document"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _relative_to_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def run() -> None:
    """Run the API with uvicorn."""
    uvicorn.run("rag_app.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
