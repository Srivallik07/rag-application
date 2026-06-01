"""Domain-specific exceptions for the RAG application."""


class RAGError(Exception):
    """Base exception for RAG application failures."""


class ConfigurationError(RAGError):
    """Raised when configuration is invalid or incomplete."""


class DocumentLoadingError(RAGError):
    """Raised when documents cannot be loaded."""


class VectorStoreError(RAGError):
    """Raised when the vector store cannot be created, loaded, or queried."""


class GenerationError(RAGError):
    """Raised when answer generation fails."""
