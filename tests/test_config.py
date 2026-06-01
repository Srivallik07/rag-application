from pathlib import Path

import pytest

from rag_app.config import AppConfig


def test_default_config_matches_required_chunking() -> None:
    config = AppConfig()

    assert config.documents.chunk_size == 1000
    assert config.documents.chunk_overlap == 200
    assert config.vector_store.top_k == 5
    assert config.vector_store.path == Path("storage/faiss_index")
    assert config.llm.provider == "openai"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "documents": {
                    "chunk_size": 1000,
                    "chunk_overlap": 1000,
                }
            }
        )
