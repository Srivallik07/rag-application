"""Build a compact JSON corpus for the Vercel serverless RAG API."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


DOCUMENT_DIR = Path("data/documents")
OUTPUT_PATH = Path("vercel_data/corpus.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def main() -> None:
    documents = []
    chunks = []
    for path in sorted(DOCUMENT_DIR.glob("**/*")):
        if not path.is_file() or path.name == ".gitkeep" or path.stat().st_size == 0:
            continue
        text_items = _load_text(path)
        if not text_items:
            continue
        documents.append(str(path))
        for item in text_items:
            for index, chunk in enumerate(_chunk_text(item["text"])):
                chunks.append(
                    {
                        "content": chunk,
                        "terms": _terms(chunk),
                        "metadata": {
                            "source": str(path),
                            "file_type": path.suffix.lower().lstrip("."),
                            "page": item.get("page"),
                            "chunk": index,
                        },
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "document_files": documents,
                "chunks": chunks,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Built {len(chunks)} chunks from {len(documents)} documents at {OUTPUT_PATH}")


def _load_text(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": page_number})
        return pages
    if suffix == ".txt":
        return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page": None}]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [{"text": json.dumps(payload, indent=2, ensure_ascii=False), "page": None}]
    return []


def _chunk_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(normalized):
        chunks.append(normalized[start : start + CHUNK_SIZE])
        start += step
    return chunks


def _terms(text: str) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(text) if len(term) > 2]


if __name__ == "__main__":
    main()
