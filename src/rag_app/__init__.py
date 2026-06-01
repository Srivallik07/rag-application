"""Naive RAG system package.

Heavy integration modules are imported lazily so lightweight modules such as
configuration and schemas can be used without importing every provider package.
"""

from rag_app.config import AppConfig, load_config

__all__ = ["AppConfig", "RAGPipeline", "load_config"]


def __getattr__(name: str) -> object:
    if name == "RAGPipeline":
        from rag_app.pipeline import RAGPipeline

        return RAGPipeline
    raise AttributeError(f"module 'rag_app' has no attribute {name!r}")
