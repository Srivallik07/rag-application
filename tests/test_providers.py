from rag_app.providers import LocalHashEmbeddings


def test_local_hash_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = LocalHashEmbeddings(dimensions=64)

    first = embeddings.embed_query("refund policy refund")
    second = embeddings.embed_query("refund policy refund")

    assert first == second
    assert len(first) == 64
    assert any(value != 0 for value in first)
