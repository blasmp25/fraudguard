"""Turn validated transactions into predictions with the served model."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd

from fraudguard.api.model_service import ModelService
from fraudguard.api.schemas import Prediction, risk_level


def score(service: ModelService, transactions: list[dict[str, Any]]) -> list[Prediction]:
    """Score a batch in a single model call; output order matches input order."""
    df = pd.DataFrame.from_records(transactions)
    probabilities = service.model.predict_proba(df)[:, 1]

    predictions = []
    for tx, p in zip(transactions, probabilities, strict=True):
        p = float(p)
        predictions.append(
            Prediction.model_validate(
                {
                    "request_id": str(uuid4()),
                    "transaction_id": tx["transaction_id"],
                    "fraud_probability": p,
                    "prediction": int(p >= service.threshold),
                    "risk_level": risk_level(p, service.threshold),
                    "model_version": service.version,
                    "threshold": service.threshold,
                }
            )
        )
    return predictions
