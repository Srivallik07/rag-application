"""Vector store lifecycle management for FAISS and Chroma."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from rag_app.config import VectorStoreConfig
from rag_app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Create, persist, load, and query the configured vector store."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings) -> None:
        self.config = config
        self.embeddings = embeddings
        self._store: Any | None = None

    @property
    def store(self) -> Any:
        if self._store is None:
            raise VectorStoreError("Vector store is not initialized. Build or load it first.")
        return self._store

    @traceable(run_type="tool", name="build_faiss_index")
    def build(self, documents: List[Document]) -> Any:
        """Build and persist an index from chunked documents."""
        if not documents:
            raise VectorStoreError("Cannot build vector store with zero documents.")

        started = perf_counter()
        try:
            if self.config.provider == "chroma":
                self._store = self._build_chroma(documents)
            else:
                self._store = FAISS.from_documents(documents, self.embeddings)
                self.save()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to build {self.config.provider} index: {exc}") from exc

        logger.info(
            "Built %s index with %d documents in %.2f ms",
            self.config.provider,
            len(documents),
            (perf_counter() - started) * 1000,
        )
        return self.store

    def save(self) -> None:
        """Persist the local FAISS index to disk."""
        if self.config.provider == "chroma":
            return
        try:
            self.config.path.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.config.path))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to save FAISS index: {exc}") from exc

    @traceable(run_type="tool", name="load_faiss_index")
    def load(self) -> Any:
        """Load an existing index."""
        if self.config.provider == "chroma":
            self._store = self._chroma_store()
            logger.info("Connected to Chroma collection %s", self.config.chroma_collection)
            return self.store

        if not self.index_exists():
            raise VectorStoreError(f"No FAISS index found at {self.config.path}")

        try:
            self._store = FAISS.load_local(
                str(self.config.path),
                self.embeddings,
                allow_dangerous_deserialization=self.config.allow_dangerous_deserialization,
            )
        except ValueError as exc:
            if "dangerous deserialization" in str(exc).lower():
                raise VectorStoreError(
                    "FAISS loading requires pickle deserialization. Set "
                    "vector_store.allow_dangerous_deserialization=true only for indexes you trust."
                ) from exc
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc

        logger.info("Loaded FAISS index from %s", self.config.path)
        return self.store

    def load_or_build(self, documents: List[Document] | None = None) -> Any:
        """Load an existing index, or build a new one when documents are supplied."""
        if self.index_exists():
            return self.load()
        if documents is None:
            raise VectorStoreError("Index does not exist and no documents were provided to build it.")
        return self.build(documents)

    def index_exists(self) -> bool:
        if self.config.provider == "chroma":
            try:
                collection = self._chroma_client().get_or_create_collection(
                    self.config.chroma_collection
                )
                return collection.count() > 0
            except VectorStoreError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Failed to inspect Chroma collection: {exc}") from exc

        path = Path(self.config.path)
        return (path / "index.faiss").exists() and (path / "index.pkl").exists()

    def _build_chroma(self, documents: List[Document]) -> Any:
        store = self._chroma_store()
        try:
            store.delete_collection()
        except Exception:  # noqa: BLE001
            logger.debug("Chroma collection did not exist before rebuild.", exc_info=True)
        store = self._chroma_store()
        store.add_documents(documents)
        return store

    def _chroma_store(self) -> Any:
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise VectorStoreError(
                "Chroma vector store requires 'chromadb' and 'langchain-chroma'. "
                "Install them with: pip install chromadb langchain-chroma"
            ) from exc

        return Chroma(
            collection_name=self.config.chroma_collection,
            embedding_function=self.embeddings,
            client=self._chroma_client(),
        )

    def _chroma_client(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "Chroma Cloud requires 'chromadb'. Install it with: pip install chromadb"
            ) from exc

        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE")
        cloud_host = os.getenv("CHROMA_HOST", self.config.chroma_host)
        if not api_key or not tenant or not database:
            raise VectorStoreError(
                "Chroma Cloud is selected, but CHROMA_API_KEY, CHROMA_TENANT, "
                "and CHROMA_DATABASE are not all set in .env."
            )

        return chromadb.CloudClient(
            cloud_host=cloud_host,
            cloud_port=443,
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
