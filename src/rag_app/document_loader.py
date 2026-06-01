"""Document loading and chunking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from rag_app.config import DocumentConfig
from rag_app.exceptions import DocumentLoadingError

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load PDF, TXT, and JSON documents from local files and split them into chunks."""

    def __init__(self, config: DocumentConfig) -> None:
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )

    @traceable(run_type="tool", name="load_local_documents")
    def load(self) -> List[Document]:
        """Load and split all configured local documents."""
        started = perf_counter()
        files = list(self._discover_files())
        if not files:
            logger.warning("No documents found under %s", self.config.input_dir)
            return []

        try:
            documents: list[Document] = []
            for file_path in files:
                loaded = self._load_file(file_path)
                documents.extend(loaded)
            chunks = self.splitter.split_documents(documents)
        except Exception as exc:  # noqa: BLE001
            raise DocumentLoadingError(f"Failed to load documents: {exc}") from exc

        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "Loaded %d files into %d chunks in %.2f ms",
            len(files),
            len(chunks),
            elapsed_ms,
        )
        return chunks

    def _discover_files(self) -> Iterable[Path]:
        input_dir = self.config.input_dir
        for pattern in self.config.glob_patterns:
            yield from sorted(input_dir.glob(pattern))

    def _load_file(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        logger.debug("Loading %s", file_path)

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
        elif suffix == ".json":
            docs = [
                Document(
                    page_content=json.dumps(
                        json.loads(file_path.read_text(encoding="utf-8")),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    metadata={"source": str(file_path), "file_type": "json"},
                )
            ]
        else:
            logger.debug("Skipping unsupported file type: %s", file_path)
            return []

        for doc in docs:
            doc.metadata.setdefault("source", str(file_path))
            doc.metadata.setdefault("file_type", suffix.lstrip("."))
        return docs
