"""Evaluate healthcare RAG retrieval accuracy and answer quality."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.config import load_config
from rag_app.logging_config import configure_logging
from rag_app.pipeline import RAGPipeline

QA_PATH = ROOT / "data" / "evaluation" / "healthcare_qa.json"
REPORT_PATH = ROOT / "data" / "evaluation" / "healthcare_eval_report.json"


def _source_text(documents: list) -> str:
    parts: list[str] = []
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        parts.append(source.lower())
        parts.append(doc.content.lower())
    return " ".join(parts)


def retrieval_hit(documents: list, expected_hint: str) -> bool:
    if expected_hint == "hc_":
        return bool(documents)
    blob = _source_text(documents)
    return expected_hint.lower() in blob


def keyword_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
    return hits / len(keywords)


def citation_score(answer: str) -> float:
    return 1.0 if re.search(r"\[source:\d+\]", answer) else 0.0


def main() -> None:
    load_dotenv(ROOT / ".env")
    config = load_config(ROOT / "config.yaml")
    configure_logging(config.logging)
    pipeline = RAGPipeline(config)

    doc_files = list(config.documents.input_dir.glob("**/*.txt"))
    if len(doc_files) < config.domain.min_source_documents:
        raise SystemExit(
            f"Expected at least {config.domain.min_source_documents} documents, found {len(doc_files)}."
        )

    print(f"Indexing {len(doc_files)} healthcare documents...")
    pipeline.index_documents(force_rebuild=True)

    cases = json.loads(QA_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    retrieval_hits = 0
    keyword_total = 0.0
    citation_total = 0.0

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        response = pipeline.answer(question, retrieval_mode="adaptive")
        hit = retrieval_hit(response.documents, case.get("expected_doc_hint", ""))
        kscore = keyword_score(response.answer, case.get("expected_keywords", []))
        cscore = citation_score(response.answer)

        retrieval_hits += int(hit)
        keyword_total += kscore
        citation_total += cscore

        results.append(
            {
                "question": question,
                "retrieval_hit": hit,
                "keyword_score": round(kscore, 3),
                "citation_score": cscore,
                "answer_preview": response.answer[:280],
                "top_source": response.documents[0].metadata.get("source") if response.documents else None,
                "latency_ms": round(response.latency_ms, 2),
            }
        )
        print(f"[{index}/{len(cases)}] hit={hit} keywords={kscore:.2f} citations={cscore:.0f}")

    n = len(cases)
    summary = {
        "domain": config.domain.name,
        "documents_indexed": len(doc_files),
        "questions_evaluated": n,
        "retrieval_accuracy": round(retrieval_hits / n, 3),
        "mean_keyword_score": round(keyword_total / n, 3),
        "citation_rate": round(citation_total / n, 3),
        "composite_quality_score": round((keyword_total + citation_total) / (2 * n), 3),
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Healthcare RAG Evaluation ===")
    print(f"Retrieval accuracy: {summary['retrieval_accuracy']:.1%}")
    print(f"Mean keyword score: {summary['mean_keyword_score']:.3f}")
    print(f"Citation rate:      {summary['citation_rate']:.1%}")
    print(f"Composite quality:  {summary['composite_quality_score']:.3f}")
    print(f"Report saved to:    {REPORT_PATH}")


if __name__ == "__main__":
    main()
