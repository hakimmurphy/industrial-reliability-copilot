from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_generates_sql_without_execution() -> None:
    payload = {"question": "Show failure rate by machine type", "execute_sql": False}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["generated_sql"]
    assert body["rows"] is None
