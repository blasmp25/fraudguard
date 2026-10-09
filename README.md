# FraudGuard

[![tests](https://github.com/blasmp25/fraudguard/actions/workflows/tests.yml/badge.svg)](https://github.com/blasmp25/fraudguard/actions/workflows/tests.yml)

**End-to-end ML platform for imbalanced fraud detection: real-time scoring API, experiment tracking, model registry, prediction logging, and (in progress) drift monitoring and gated retraining.**

Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590k transactions, 3.5% fraud).

> 🚧 Built in public, one phase at a time. The MVP (phases 0–2) is complete.

## Status

- [x] Phase 0 — Project skeleton and tooling (uv, ruff, mypy, pre-commit, pytest, CI)
- [x] Phase 1 — Data, temporal split, feature selection, baselines tracked in MLflow
- [x] Phase 2 — FastAPI scoring service, prediction logging, Docker (**MVP**)
- [ ] Phase 3 — Imbalance-handling study (class weights, undersampling, SMOTE variants)
- [ ] Phase 4 — Deployment on GCP (Cloud Run, BigQuery, Cloud Storage)
- [ ] Phase 5 — Event-driven ingestion with Pub/Sub + traffic simulator
- [ ] Phase 6 — Drift and performance monitoring
- [ ] Phase 7 — Gated retraining (champion / challenger)
- [ ] Phase 8 — Documentation and write-up

## Architecture (MVP)

```
 Kaggle CSVs ──► make_dataset ──► Parquet ──► train_baselines ──► MLflow
                                  (temporal split)                 runs + registry
                                                                   fraudguard@champion
                                                                        │
                                                                 export_champion
                                                                        ▼
 Client ──HTTP──► FastAPI (Docker) ──► preprocessing + LightGBM ──► decision
                       │                 (one serialized pipeline)
                       └──── background task ────► prediction log (SQLite)
```

- **Training side:** data is split by time, models are tracked in MLflow, and the best one is registered as `fraudguard@champion` together with its decision threshold.
- **Serving side:** the champion is exported and baked into a Docker image. The API validates each request, scores it, applies the threshold, and logs the prediction without slowing down the response.

## Results (validation set, days 120–149)

| Model | PR-AUC | ROC-AUC | Recall @ precision 0.5 | Train time |
| --- | --- | --- | --- | --- |
| **LightGBM (champion)** | **0.609** | 0.925 | 0.607 | 26 s |
| XGBoost | 0.606 | 0.926 | 0.605 | 45 s |
| Random Forest | 0.522 | 0.909 | 0.501 | 171 s |
| Logistic Regression | 0.399 | 0.830 | 0.342 | 51 s |

At the chosen threshold (0.185), **65% of alerts are real fraud and 53% of fraud is caught**. LightGBM and XGBoost are tied within run-to-run noise; LightGBM was chosen for training speed. Details in [docs/experiments.md](docs/experiments.md).

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness check |
| `GET /model` | Served model name, version, threshold and expected features |
| `POST /predict` | Score one transaction |
| `POST /predict/batch` | Score up to 1,000 transactions in one call |

Only `transaction_id`, `TransactionDT` and `TransactionAmt` are required; the other 57 features may be `null`, as they often are in real data. Unknown fields are rejected with a 422.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 1, "TransactionDT": 13000000, "TransactionAmt": 120.5, "P_emaildomain": "gmail.com"}'
```

```json
{
  "request_id": "a467cb11-b845-4595-89e0-be97727060c3",
  "transaction_id": 1,
  "fraud_probability": 0.0412,
  "prediction": 0,
  "risk_level": "LOW",
  "model_version": "1",
  "threshold": 0.185039
}
```

`risk_level` is `HIGH` at or above the threshold (block / manual review), `MEDIUM` at or above half the threshold (step-up verification) and `LOW` otherwise. Interactive docs are served at `/docs`.

## Reproduce it

Requirements: [uv](https://docs.astral.sh/uv/), Docker, and a Kaggle account with the competition rules accepted (see [data/README.md](data/README.md)).

```bash
uv sync                                        # install everything
uv run python scripts/download_data.py         # ~60 MB download from Kaggle
uv run python scripts/make_dataset.py          # merge + Parquet
uv run python scripts/train_baselines.py       # train 4 models, register the champion
uv run python scripts/export_champion.py       # export it for packaging
docker compose up --build                      # API on http://127.0.0.1:8000
```

Inspect experiments with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Design decisions

- **Temporal split, never random.** Card-level fields identify customers, so a random split leaks the same cards into train and test and inflates metrics. Train on days < 120, validate on 120–149, keep 150+ as unseen "production" traffic.
- **PR-AUC as the main metric.** At 3.5% prevalence, ROC-AUC barely separates models (0.83–0.93) while PR-AUC does (0.40–0.61).
- **The threshold is part of the model.** It is chosen on validation and stored as a tag on the registered model version, so the API never hard-codes it.
- **One pipeline from training to serving.** Preprocessing (category handling, rare-category grouping, `hour` feature) is a scikit-learn transformer serialized together with the model, which rules out training/serving skew.
- **Missing values are kept, not imputed, for tree models.** EDA showed that missingness is structural and predictive (e.g. identity data is absent by design for one product type).
- **The request schema is generated from the served model.** The API reads the expected features from the loaded pipeline, so schema and model can never drift apart. Unknown fields fail loudly instead of becoming silent nulls.
- **The model is baked into the image.** The image is fully reproducible (code + model + locked dependencies) and starts without access to the registry.
- **Serving and training dependencies are split.** The production image only installs what the API needs: 2.5 GB → 707 MB. Note: the champion's library (currently LightGBM) must stay in the serving dependencies.
- **Prediction logging never affects the client.** Logs are written in a background task after the response; a database failure is logged, not propagated.
- **Security basics.** The container runs as a non-root user, internal errors never reach the client, and models are serialized with pickle only because they are loaded exclusively from our own registry.

## Project structure

```
src/fraudguard/
├── config/       settings loaded from configs/*.yaml and validated with pydantic
├── data/         loading, Parquet build, temporal split
├── features/     shared preprocessing (FraudPreprocessor)
├── models/       baseline pipelines and evaluation metrics
└── api/          FastAPI app, request/response schemas, scoring, prediction log
scripts/          download, dataset build, training, export
notebooks/        EDA, feature selection, champion check
tests/            unit and API tests (run in CI without data or MLflow)
```

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

Every commit runs ruff, formatting, mypy and the tests (pre-commit locally, GitHub Actions on push).

## Data and license

The dataset belongs to Vesta Corporation / Kaggle and is not redistributed here. Code released under the MIT License.
