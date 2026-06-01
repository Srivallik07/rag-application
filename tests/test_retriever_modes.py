from langchain_core.documents import Document

from rag_app.config import RetrievalConfig, VectorStoreConfig
from rag_app.retriever import Retriever


class FakeVectorStore:
    def __init__(self, scored_docs):
        self.scored_docs = scored_docs
        self.queries = []

    def similarity_search_with_score(self, query: str, k: int):
        self.queries.append((query, k))
        return self.scored_docs[:k]


def test_advanced_retrieval_diversifies_and_marks_mode() -> None:
    store = FakeVectorStore(
        [
            (Document(page_content="refund policy refund window", metadata={"source": "a.txt"}), 0.1),
            (Document(page_content="refund policy duplicate wording", metadata={"source": "b.txt"}), 0.2),
            (Document(page_content="exchange rules and store credit", metadata={"source": "c.txt"}), 0.3),
        ]
    )
    retriever = Retriever(
        store,
        VectorStoreConfig(top_k=2),
        RetrievalConfig(mode="advanced", advanced_fetch_multiplier=2),
    )

    result = retriever.retrieve("refund policy", top_k=2)

    assert result.mode == "advanced"
    assert len(result.documents) == 2
    assert store.queries == [("refund policy", 4)]
    assert all(document.metadata["retrieval_mode"] == "advanced" for document in result.documents)


def test_corrective_retrieval_rewrites_weak_query() -> None:
    store = FakeVectorStore(
        [
            (Document(page_content="pandas dataframe merge examples", metadata={"source": "pandas.txt"}), 0.6),
            (Document(page_content="series indexing operations", metadata={"source": "series.txt"}), 0.7),
        ]
    )
    retriever = Retriever(
        store,
        VectorStoreConfig(top_k=1),
        RetrievalConfig(mode="corrective", corrective_min_overlap=1),
    )

    result = retriever.retrieve("what is completely unrelated", top_k=1)

    assert result.mode == "corrective"
    assert result.rewritten_query == "completely unrelated"
    assert store.queries[0] == ("what is completely unrelated", 1)
    assert store.queries[1] == ("completely unrelated", 4)
    assert result.documents[0].metadata["retrieval_status"] == "retried"


def test_adaptive_retrieval_routes_heuristically() -> None:
    store = FakeVectorStore(
        [
            (Document(page_content="refund policy refund window", metadata={"source": "a.txt"}), 0.1),
            (Document(page_content="pandas dataframe merge examples", metadata={"source": "pandas.txt"}), 0.6),
        ]
    )
    retriever = Retriever(
        store,
        VectorStoreConfig(top_k=2),
        RetrievalConfig(mode="adaptive"),
        llm=None,
    )

    # Simple/short query should route to naive
    result_naive = retriever.retrieve("refund window", top_k=1)
    assert result_naive.mode == "adaptive"
    assert result_naive.routed_mode == "naive"
    assert len(result_naive.documents) == 1

    # Query with comparison keywords should route to advanced
    result_advanced = retriever.retrieve("compare refund and pandas", top_k=1)
    assert result_advanced.mode == "adaptive"
    assert result_advanced.routed_mode == "advanced"

    # Query with corrective keywords should route to corrective
    result_corrective = retriever.retrieve("what is wrong with pandas merge?", top_k=1)
    assert result_corrective.mode == "adaptive"
    assert result_corrective.routed_mode == "corrective"


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.invoked_prompts = []

    def invoke(self, messages, config=None):
        self.invoked_prompts.append(messages)
        # Mock class matching langchain Message response
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.response_text)


def test_adaptive_retrieval_routes_via_llm() -> None:
    store = FakeVectorStore(
        [
            (Document(page_content="some content", metadata={"source": "a.txt"}), 0.1),
        ]
    )
    fake_llm = FakeLLM("advanced")
    retriever = Retriever(
        store,
        VectorStoreConfig(top_k=1),
        RetrievalConfig(mode="adaptive"),
        llm=fake_llm,
    )

    result = retriever.retrieve("custom complex question", top_k=1)
    assert result.mode == "adaptive"
    assert result.routed_mode == "advanced"
    assert len(fake_llm.invoked_prompts) == 1
    # Check that prompt contains our instructions
    prompt_content = fake_llm.invoked_prompts[0][0].content
    assert "query router" in prompt_content
    assert "custom complex question" in prompt_content
