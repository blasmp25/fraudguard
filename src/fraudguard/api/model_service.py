"""Load the served model from the MLflow registry, with everything the API needs about it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraudguard.config import Settings
from fraudguard.features.preprocessing import FeatureSpec


class ProbabilisticModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ModelService:
    model: ProbabilisticModel
    spec: FeatureSpec
    name: str
    alias: str
    version: str
    threshold: float

    @property
    def feature_names(self) -> list[str]:
        return [*self.spec.numeric, *self.spec.categorical]


def load_champion(settings: Settings) -> ModelService:
    """Resolve the alias to a version once, then load that exact version and its threshold."""
    cfg = settings.mlflow
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(cfg.registered_model, cfg.serving_alias)
    model = mlflow.sklearn.load_model(f"models:/{cfg.registered_model}/{version.version}")

    return ModelService(
        model=model,
        spec=model.named_steps["prep"].spec,
        name=cfg.registered_model,
        alias=cfg.serving_alias,
        version=str(version.version),
        threshold=float(version.tags["threshold"]),
    )
