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


class CvConfig(BaseModel):
    """Expanding-window temporal folds: validation windows start at valid_starts."""

    valid_starts: list[int] = [60, 90, 120]
    valid_days: int = 30
    gap_days: int = 7

    @model_validator(mode="after")
    def check_windows(self) -> CvConfig:
        starts = self.valid_starts
        if not starts or any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
            raise ValueError("valid_starts must be non-empty and strictly increasing")
        if self.valid_days <= 0 or self.gap_days < 0:
            raise ValueError("valid_days must be > 0 and gap_days >= 0")
        if starts[0] - self.gap_days <= 1:
            raise ValueError("The first fold would have no training data")
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
    model_dir: Path | None = None  # if set, load an exported model instead of the registry

    @field_validator("predictions_db", "model_dir")
    @classmethod
    def make_absolute(cls, value: Path | None) -> Path | None:
        if value is None or value.is_absolute():
            return value
        return PROJECT_ROOT / value


class Settings(BaseModel):
    data: DataConfig
    split: SplitConfig
    cv: CvConfig = CvConfig()
    mlflow: MlflowConfig = MlflowConfig()
    serving: ServingConfig = ServingConfig()
    seed: int = 42

    @model_validator(mode="after")
    def folds_never_touch_future(self) -> Settings:
        last_valid_end = self.cv.valid_starts[-1] + self.cv.valid_days
        if last_valid_end > self.split.valid_end_day:
            raise ValueError(
                f"CV folds reach day {last_valid_end}, beyond valid_end_day "
                f"{self.split.valid_end_day}: they would use the future set"
            )
        return self


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from `path`, the FRAUDGUARD_CONFIG env var, or configs/local.yaml."""
    if path is None:
        path = os.environ.get("FRAUDGUARD_CONFIG") or DEFAULT_CONFIG
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
