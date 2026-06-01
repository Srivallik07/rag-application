"""Index the backend healthcare corpus (55+ documents)."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.config import load_config
from rag_app.logging_config import configure_logging
from rag_app.pipeline import RAGPipeline


def main() -> None:
    load_dotenv(ROOT / ".env")
    config = load_config(ROOT / "config.yaml")
    configure_logging(config.logging)
    files = list(config.documents.input_dir.glob("**/*.txt"))
    print(f"Found {len(files)} healthcare source files.")
    pipeline = RAGPipeline(config)
    chunks = pipeline.index_documents(force_rebuild=True)
    print(f"Indexed {chunks} document chunks into {config.vector_store.path}")


if __name__ == "__main__":
    main()
