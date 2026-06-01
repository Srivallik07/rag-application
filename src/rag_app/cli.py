"""Command-line interface for indexing documents and asking questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv

from rag_app.config import load_config
from rag_app.logging_config import configure_logging
from rag_app.pipeline import RAGPipeline

app = typer.Typer(help="Production-ready Adaptive RAG system.")


def _pipeline(config_path: Path) -> RAGPipeline:
    load_dotenv()
    config = load_config(config_path)
    configure_logging(config.logging)
    return RAGPipeline(config)


@app.command()
def index(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    force: bool = typer.Option(False, "--force", help="Rebuild the FAISS index even if it exists."),
) -> None:
    """Index configured documents into FAISS."""
    pipeline = _pipeline(config)
    chunk_count = pipeline.index_documents(force_rebuild=force)
    typer.echo(f"Indexed {chunk_count} chunks.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from indexed documents."),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k"),
    retrieval_mode: Optional[Literal["naive", "advanced", "corrective", "adaptive"]] = typer.Option(
        None,
        "--retrieval-mode",
        "-m",
        help="Retrieval strategy to use for this question.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full response as JSON."),
) -> None:
    """Ask a question against the indexed documents."""
    pipeline = _pipeline(config)
    response = pipeline.answer(question, top_k=top_k, retrieval_mode=retrieval_mode)

    if json_output:
        typer.echo(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return

    typer.echo(response.answer)
    typer.echo("\nSources:")
    for document in response.documents:
        source = document.metadata.get("source", "unknown")
        typer.echo(f"- [{document.rank}] score={document.score:.6f} source={source}")


if __name__ == "__main__":
    app()
