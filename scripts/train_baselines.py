"""Train baseline models on the temporal split and track them in MLflow.

The best model by validation PR-AUC is registered as `fraudguard@champion`,
with its decision threshold stored as a model-version tag.

Usage (from the repository root):
    uv run python scripts/train_baselines.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from sklearn.metrics import PrecisionRecallDisplay

from fraudguard.config import load_settings
from fraudguard.data.load import PROCESSED_FILE, load_dataset
from fraudguard.data.split import temporal_split
from fraudguard.features.preprocessing import FeatureSpec
from fraudguard.models.baselines import MODEL_NAMES, build_pipeline
from fraudguard.models.metrics import EvaluationResult, evaluate

REGISTERED_MODEL = "fraudguard"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    settings = load_settings()
    spec = FeatureSpec.from_yaml()
    split = temporal_split(load_dataset(settings), settings.split)
    y_train, y_valid = split.train["isFraud"], split.valid["isFraud"]
    data_md5 = file_md5(settings.data.processed_dir / PROCESSED_FILE)

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    results: list[tuple[str, EvaluationResult, str]] = []
    for name in MODEL_NAMES:
        print(f"\n=== {name} ===")
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(name, spec, seed=settings.seed)

            start = time.perf_counter()
            pipeline.fit(split.train, y_train)
            train_seconds = time.perf_counter() - start

            proba = pipeline.predict_proba(split.valid)[:, 1]
            metrics = evaluate(y_valid, proba)

            mlflow.set_tags({"stage": "baseline", "data_md5": data_md5})
            mlflow.log_params(
                {
                    "model": name,
                    "n_features": len(pipeline.named_steps["prep"].output_columns),
                    "min_category_count": pipeline.named_steps["prep"].min_category_count,
                    "train_end_day": settings.split.train_end_day,
                    "valid_end_day": settings.split.valid_end_day,
                    "seed": settings.seed,
                }
            )
            model_params = pipeline.named_steps["model"].get_params()
            mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})
            mlflow.log_metrics({**metrics.as_dict(), "train_seconds": train_seconds})
            mlflow.log_dict(
                {"numeric": spec.numeric, "categorical": spec.categorical}, "features.json"
            )

            fig = Figure(figsize=(6, 5))
            ax = fig.subplots()
            PrecisionRecallDisplay.from_predictions(y_valid, proba, ax=ax, name=name)
            mlflow.log_figure(fig, "pr_curve_valid.png")

            # cloudpickle: models are only ever loaded from our own registry (trusted source)
            info = mlflow.sklearn.log_model(
                pipeline, name="model", serialization_format="cloudpickle"
            )
            results.append((name, metrics, info.model_uri))
            print(
                f"PR-AUC {metrics.pr_auc:.4f} | recall@P50 {metrics.recall_at_p50:.3f} "
                f"| {train_seconds:.0f}s"
            )

    best_name, best_metrics, best_uri = max(results, key=lambda r: r[1].pr_auc)
    version = mlflow.register_model(best_uri, REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(
        REGISTERED_MODEL, version.version, "threshold", f"{best_metrics.threshold:.6f}"
    )

    table = pd.DataFrame({name: m.as_dict() for name, m, _ in results}).T
    print("\n", table.round(4).sort_values("pr_auc", ascending=False))
    print(
        f"\nChampion: {best_name} -> {REGISTERED_MODEL} v{version.version} "
        f"(threshold {best_metrics.threshold:.4f})"
    )


if __name__ == "__main__":
    main()
