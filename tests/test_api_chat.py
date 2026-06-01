from fastapi.testclient import TestClient

from rag_app.api import app


def test_chat_requires_user_message() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "assistant", "content": "Hello"}]},
    )

    assert response.status_code == 400
