from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from fraudguard.api.app import create_app
from fraudguard.api.model_service import ModelService
from fraudguard.features.preprocessing import FeatureSpec

API_SPEC = FeatureSpec(numeric=["TransactionAmt", "C1"], categorical=["P_emaildomain"])


class FakeModel:
    """Stands in for the real pipeline: always returns the same fraud probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        p = np.full(len(X), self.probability)
        return np.column_stack([1 - p, p])


def build_client(
    probability: float = 0.9, threshold: float = 0.2, model: object | None = None
) -> TestClient:
    service = ModelService(
        model=model or FakeModel(probability),  # type: ignore[arg-type]
        spec=API_SPEC,
        name="fraudguard",
        alias="champion",
        version="7",
        threshold=threshold,
    )
    return TestClient(create_app(service))


@pytest.fixture
def client() -> TestClient:
    return build_client()


@pytest.fixture
def client_factory() -> Callable[..., TestClient]:
    """Lets a test build a client with its own probability, threshold or model."""
    return build_client
