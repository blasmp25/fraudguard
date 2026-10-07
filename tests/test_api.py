from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from fraudguard.api.schemas import MAX_BATCH_SIZE

TX = {"transaction_id": 42, "TransactionDT": 86_400, "TransactionAmt": 50.0}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["version"] == "7"
    assert body["threshold"] == 0.2
    assert body["features"] == ["TransactionAmt", "C1", "P_emaildomain"]


@pytest.mark.parametrize(
    ("probability", "expected_risk", "expected_prediction"),
    [(0.9, "HIGH", 1), (0.2, "HIGH", 1), (0.15, "MEDIUM", 0), (0.05, "LOW", 0)],
)
def test_predict_applies_threshold(
    client_factory: Callable[..., TestClient],
    probability: float,
    expected_risk: str,
    expected_prediction: int,
) -> None:
    client = client_factory(probability=probability, threshold=0.2)
    body = client.post("/predict", json=TX).json()
    assert body["risk_level"] == expected_risk
    assert body["prediction"] == expected_prediction


def test_predict_echoes_ids_and_model(client: TestClient) -> None:
    body = client.post("/predict", json=TX).json()
    assert body["transaction_id"] == 42
    assert body["model_version"] == "7"
    assert body["threshold"] == 0.2
    assert body["request_id"]


def test_typo_in_field_returns_422(client: TestClient) -> None:
    payload = {**TX, "TransactionAmnt": 50.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_amount_returns_422(client: TestClient) -> None:
    payload = {k: v for k, v in TX.items() if k != "TransactionAmt"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_keeps_order_and_unique_request_ids(client: TestClient) -> None:
    batch = [{**TX, "transaction_id": i} for i in range(5)]
    body = client.post("/predict/batch", json=batch).json()
    assert [p["transaction_id"] for p in body] == [0, 1, 2, 3, 4]
    assert len({p["request_id"] for p in body}) == 5


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_limits(client: TestClient, size: int) -> None:
    batch = [{**TX, "transaction_id": i} for i in range(size)]
    assert client.post("/predict/batch", json=batch).status_code == 422


class FailingModel:
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        raise RuntimeError("internal detail that must not leak")


class NanModel:
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return np.array([[np.nan, np.nan]])


@pytest.mark.parametrize("model", [FailingModel(), NanModel()])
def test_model_errors_return_generic_500(
    client_factory: Callable[..., TestClient], model: object
) -> None:
    response = client_factory(model=model).post("/predict", json=TX)
    assert response.status_code == 500
    assert "internal detail" not in response.text


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list = []

    def log(self, records: list) -> None:
        self.records.extend(records)


class BrokenLogger:
    def log(self, records: list) -> None:
        raise RuntimeError("database is down")


def test_every_prediction_is_logged_with_its_input(
    client_factory: Callable[..., TestClient],
) -> None:
    recorder = RecordingLogger()
    batch = [{**TX, "transaction_id": i} for i in range(3)]
    body = client_factory(prediction_logger=recorder).post("/predict/batch", json=batch).json()

    assert [r.request_id for r in recorder.records] == [p["request_id"] for p in body]
    assert [r.features["transaction_id"] for r in recorder.records] == [0, 1, 2]
    assert all(r.model_version == "7" for r in recorder.records)


def test_logging_failure_does_not_break_prediction(
    client_factory: Callable[..., TestClient],
) -> None:
    response = client_factory(prediction_logger=BrokenLogger()).post("/predict", json=TX)
    assert response.status_code == 200
