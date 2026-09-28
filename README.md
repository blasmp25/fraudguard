# FraudGuard

**End-to-end ML platform for imbalanced fraud detection, real-time inference, model monitoring and automated retraining.**

> 🚧 Work in progress — built in public, one phase at a time.

## Status

- [x] Phase 0 — Project skeleton, tooling (uv, ruff, mypy, pre-commit, pytest)
- [ ] Phase 1 — Data, temporal split, baselines tracked in MLflow
- [ ] Phase 2 — FastAPI service, Docker, CI (**MVP**)
- [ ] Phase 3 — Imbalance-handling study (SMOTE variants, class weights, undersampling)
- [ ] Phase 4 — Deployment on GCP (Cloud Run, BigQuery, Cloud Storage)
- [ ] Phase 5 — Event-driven ingestion with Pub/Sub + traffic simulator
- [ ] Phase 6 — Drift and performance monitoring
- [ ] Phase 7 — Gated retraining (champion/challenger)
- [ ] Phase 8 — Documentation and write-up

## Planned architecture

```
Transaction → Pub/Sub → Cloud Run (FastAPI) → BigQuery
                                 ↑                │
                         MLflow Registry     Monitoring (drift, PR-AUC)
                                 ↑                │
                            Retraining  ←─────────┘
```

## Tech stack

Python 3.11 · LightGBM · scikit-learn · imbalanced-learn · MLflow · FastAPI · Docker · GitHub Actions · GCP

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

MIT
