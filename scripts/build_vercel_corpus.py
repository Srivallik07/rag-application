"""Build Vercel corpus from healthcare documents and sync frontend to public/."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_DIR = ROOT / "data" / "documents" / "healthcare"
OUTPUT_PATH = ROOT / "vercel_data" / "corpus.json"
PUBLIC_DIR = ROOT / "public"
FRONTEND_DIR = ROOT / "frontend"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def main() -> None:
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 500:
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("chunks"):
            _sync_public()
            print(
                f"Using bundled corpus: {len(existing['chunks'])} chunks, "
                f"{len(existing.get('document_files', []))} documents."
            )
            return

    if not DOCUMENT_DIR.exists():
        raise SystemExit(f"Healthcare corpus directory not found: {DOCUMENT_DIR}")

    documents: list[str] = []
    chunks: list[dict[str, Any]] = []

    for path in sorted(DOCUMENT_DIR.glob("**/*")):
        if not path.is_file() or path.name in {".gitkeep", "corpus_manifest.json"}:
            continue
        if path.stat().st_size == 0:
            continue
        text_items = _load_text(path)
        if not text_items:
            continue
        rel_path = str(path.relative_to(ROOT))
        documents.append(rel_path)
        for item in text_items:
            for index, chunk in enumerate(_chunk_text(item["text"])):
                chunks.append(
                    {
                        "content": chunk,
                        "terms": _terms(chunk),
                        "metadata": {
                            "source": rel_path,
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
                "domain": "healthcare",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Built {len(chunks)} chunks from {len(documents)} healthcare documents.")
    _sync_public()


def _sync_public() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    if not FRONTEND_DIR.exists():
        print("Frontend folder not in deployment; using committed public/ assets.")
        return
    for name in ("index.html", "styles.css", "app.js"):
        source = FRONTEND_DIR / name
        if source.exists():
            shutil.copy2(source, PUBLIC_DIR / name)


def _load_text(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

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
    result: list[str] = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(normalized):
        result.append(normalized[start : start + CHUNK_SIZE])
        start += step
    return result


def _terms(text: str) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(text) if len(term) > 2]


if __name__ == "__main__":
    main()
