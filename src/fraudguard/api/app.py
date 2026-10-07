"""FastAPI application. It receives an already-loaded model (see main.py).

Note: no `from __future__ import annotations` here. FastAPI reads the type
annotations at runtime to know what to validate, and the request schema is
built inside create_app from the served model's features.
"""

import logging
from typing import Any, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from fraudguard import __version__
from fraudguard.api.model_service import ModelService
from fraudguard.api.prediction_log import PredictionLogger, build_records, log_safely
from fraudguard.api.schemas import (
    MAX_BATCH_SIZE,
    Health,
    ModelInfo,
    Prediction,
    build_transaction_model,
)
from fraudguard.api.scoring import score

logger = logging.getLogger(__name__)


def create_app(service: ModelService, prediction_logger: PredictionLogger) -> FastAPI:
    app = FastAPI(
        title="FraudGuard",
        version=__version__,
        description="Real-time fraud scoring API",
    )
    # A class built at runtime from the served model's features, hence CapWords
    Transaction = build_transaction_model(service.spec)  # noqa: N806

    def score_or_500(records: list[dict[str, Any]]) -> list[Prediction]:
        try:
            return score(service, records)
        except Exception:
            logger.exception("Scoring failed")  # full details stay in our logs
            raise HTTPException(status_code=500, detail="Model failed to score") from None

    @app.get("/health", response_model=Health)
    def health() -> Health:
        """Liveness check: the process is up and the model is loaded."""
        return Health(status="ok")

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        """Which model version is being served, with which threshold and features."""
        return ModelInfo(
            name=service.name,
            alias=service.alias,
            version=service.version,
            threshold=service.threshold,
            features=service.feature_names,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction, tasks: BackgroundTasks) -> Prediction:  # type: ignore[valid-type]
        """Score one transaction."""
        inputs = [cast(BaseModel, tx).model_dump()]
        predictions = score_or_500(inputs)
        records = build_records(predictions, inputs, service.name)
        tasks.add_task(log_safely, prediction_logger, records)
        return predictions[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(txs: list[Transaction], tasks: BackgroundTasks) -> list[Prediction]:  # type: ignore[valid-type]
        """Score up to MAX_BATCH_SIZE transactions; output order matches input order."""
        if not 1 <= len(txs) <= MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=422, detail=f"Batch size must be between 1 and {MAX_BATCH_SIZE}"
            )
        inputs = [cast(BaseModel, t).model_dump() for t in txs]
        predictions = score_or_500(inputs)
        records = build_records(predictions, inputs, service.name)
        tasks.add_task(log_safely, prediction_logger, records)
        return predictions

    return app
