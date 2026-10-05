from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["version"] == "7"
    assert body["threshold"] == 0.2
    assert body["features"] == ["TransactionAmt", "C1", "P_emaildomain"]
