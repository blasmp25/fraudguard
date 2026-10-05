"""Request/response contract of the prediction API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraudguard.features.preprocessing import FeatureSpec

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
MEDIUM_RISK_FACTOR = 0.5  # MEDIUM if probability >= threshold * this factor
MAX_BATCH_SIZE = 1000


def build_transaction_model(spec: FeatureSpec) -> type[BaseModel]:
    """Build the request schema from the feature spec of the model being served."""
    fields: dict[str, Any] = {
        "transaction_id": (int, Field(ge=0, description="Client-side id, used to join labels")),
        "TransactionDT": (int, Field(ge=0, description="Seconds from the reference time")),
        "TransactionAmt": (float, Field(gt=0, description="Transaction amount")),
    }
    for col in spec.numeric:
        fields.setdefault(col, (float | None, None))
    for col in spec.categorical:
        fields.setdefault(col, (str | None, None))
    return create_model("Transaction", __config__=ConfigDict(extra="forbid"), **fields)


def risk_level(probability: float, threshold: float) -> RiskLevel:
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * MEDIUM_RISK_FACTOR:
        return "MEDIUM"
    return "LOW"


class Prediction(BaseModel):
    request_id: str
    transaction_id: int
    fraud_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    prediction: Literal[0, 1]
    risk_level: RiskLevel
    model_version: str
    threshold: float


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str
    threshold: float
    features: list[str]


class Health(BaseModel):
    status: Literal["ok"]
