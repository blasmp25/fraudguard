"""Project configuration loaded from a YAML file and validated with pydantic."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "local.yaml"


class DataConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path

    @field_validator("raw_dir", "processed_dir")
    @classmethod
    def make_absolute(cls, value: Path) -> Path:
        """Resolve relative paths against the project root."""
        return value if value.is_absolute() else PROJECT_ROOT / value


class SplitConfig(BaseModel):
    train_end_day: int
    valid_end_day: int

    @model_validator(mode="after")
    def check_order(self) -> SplitConfig:
        if not 0 < self.train_end_day < self.valid_end_day:
            raise ValueError("Require 0 < train_end_day < valid_end_day")
        return self


class MlflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "fraudguard"
    registered_model: str = "fraudguard"
    serving_alias: str = "champion"

    @field_validator("tracking_uri")
    @classmethod
    def make_sqlite_path_absolute(cls, value: str) -> str:
        """Anchor relative SQLite paths at the project root (so notebooks find the same DB)."""
        prefix = "sqlite:///"
        if value.startswith(prefix) and not Path(value[len(prefix) :]).is_absolute():
            return prefix + (PROJECT_ROOT / value[len(prefix) :]).as_posix()
        return value


class ServingConfig(BaseModel):
    predictions_db: Path = Path("data/predictions.db")

    @field_validator("predictions_db")
    @classmethod
    def make_absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value


class Settings(BaseModel):
    data: DataConfig
    split: SplitConfig
    mlflow: MlflowConfig = MlflowConfig()
    serving: ServingConfig = ServingConfig()
    seed: int = 42


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from `path`, the FRAUDGUARD_CONFIG env var, or configs/local.yaml."""
    if path is None:
        path = os.environ.get("FRAUDGUARD_CONFIG") or DEFAULT_CONFIG
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
