"""Entry point: load the champion from the registry and build the app.

Usage (from the repository root):
    uv run uvicorn fraudguard.api.main:build_app --factory --reload
"""

from fastapi import FastAPI

from fraudguard.api.app import create_app
from fraudguard.api.model_service import load_champion, load_exported
from fraudguard.api.prediction_log import SQLitePredictionLogger
from fraudguard.config import load_settings


def build_app() -> FastAPI:
    settings = load_settings()
    model_dir = settings.serving.model_dir
    service = load_exported(model_dir) if model_dir else load_champion(settings)
    return create_app(service, SQLitePredictionLogger(settings.serving.predictions_db))
