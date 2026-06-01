from rag_app.llm_chain import LLMChain
from rag_app.prompting import PromptBuilder
from rag_app.schemas import RetrievedDocument


def test_extractive_answer_uses_retrieved_documents() -> None:
    chain = LLMChain(None, PromptBuilder(), "extractive-local", provider="extractive")
    docs = [
        RetrievedDocument(
            content="Artificial intelligence is a field of computer science. It builds systems that can perform tasks associated with human intelligence.",
            metadata={"source": "rag.txt"},
            score=0.1,
            rank=1,
        )
    ]

    result = chain.generate("what is artificial intelligence", docs)

    assert "Artificial intelligence" in result.answer
    assert "[source:1]" in result.answer
