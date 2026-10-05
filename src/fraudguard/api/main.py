"""Entry point: load the champion from the registry and build the app.

Usage (from the repository root):
    uv run uvicorn fraudguard.api.main:build_app --factory --reload
"""

from fastapi import FastAPI

from fraudguard.api.app import create_app
from fraudguard.api.model_service import load_champion
from fraudguard.config import load_settings


def build_app() -> FastAPI:
    return create_app(load_champion(load_settings()))
