"""FastAPI application. It receives an already-loaded model (see main.py)."""

from __future__ import annotations

from fastapi import FastAPI

from fraudguard import __version__
from fraudguard.api.model_service import ModelService
from fraudguard.api.schemas import Health, ModelInfo


def create_app(service: ModelService) -> FastAPI:
    app = FastAPI(
        title="FraudGuard",
        version=__version__,
        description="Real-time fraud scoring API",
    )

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

    return app
