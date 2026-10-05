import pytest
from pydantic import ValidationError

from fraudguard.api.schemas import Prediction, build_transaction_model, risk_level
from fraudguard.features.preprocessing import FeatureSpec

SPEC = FeatureSpec(numeric=["TransactionAmt", "C1"], categorical=["P_emaildomain"])
Transaction = build_transaction_model(SPEC)
REQUIRED = {"transaction_id": 1, "TransactionDT": 86_400, "TransactionAmt": 50.0}


def test_only_required_fields_is_valid() -> None:
    tx = Transaction(**REQUIRED)
    assert tx.C1 is None  # optional features default to null


def test_missing_amount_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Transaction(transaction_id=1, TransactionDT=86_400)


def test_unknown_field_is_rejected() -> None:
    """A typo must fail loudly instead of silently becoming a null feature."""
    with pytest.raises(ValidationError):
        Transaction(**REQUIRED, TransactionAmnt=50.0)


def test_non_positive_amount_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Transaction(**{**REQUIRED, "TransactionAmt": 0})


def test_risk_levels() -> None:
    assert risk_level(0.20, threshold=0.2) == "HIGH"
    assert risk_level(0.10, threshold=0.2) == "MEDIUM"
    assert risk_level(0.09, threshold=0.2) == "LOW"


def test_nan_probability_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Prediction(
            request_id="r",
            transaction_id=1,
            fraud_probability=float("nan"),
            prediction=0,
            risk_level="LOW",
            model_version="1",
            threshold=0.2,
        )
