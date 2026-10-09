"""Load the served model from the MLflow registry, with everything the API needs about it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
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
    import mlflow
    import mlflow.sklearn  # lazy: only needed when loading from the registry (not in the image)

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


def export_model(service: ModelService, out_dir: Path) -> None:
    """Write the served model and its metadata to a folder, for packaging in an image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(service.model, out_dir / "model.joblib")
    metadata = {
        "name": service.name,
        "alias": service.alias,
        "version": service.version,
        "threshold": service.threshold,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_exported(model_dir: Path) -> ModelService:
    """Load a model exported with export_model (no MLflow registry needed)."""
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "model.joblib")
    return ModelService(
        model=model,
        spec=model.named_steps["prep"].spec,
        name=metadata["name"],
        alias=metadata["alias"],
        version=str(metadata["version"]),
        threshold=float(metadata["threshold"]),
    )
