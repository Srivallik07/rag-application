import pytest

pytest.importorskip("langchain_core")

from rag_app.prompting import PromptBuilder
from rag_app.schemas import RetrievedDocument


def test_prompt_contains_query_context_and_source_rank() -> None:
    builder = PromptBuilder()
    document = RetrievedDocument(
        content="Refunds are available within 30 days.",
        metadata={"source": "policy.txt"},
        score=0.12,
        rank=1,
    )

    prompt = builder.render_prompt("What is the refund policy?", [document], chat_history="user: Hi")

    assert "What is the refund policy?" in prompt
    assert "Refunds are available within 30 days." in prompt
    assert "[source:1]" in prompt
    assert "user: Hi" in prompt
