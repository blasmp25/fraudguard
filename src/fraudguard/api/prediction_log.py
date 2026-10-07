"""Persist every prediction so it can be monitored and joined with labels later."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fraudguard.api.schemas import Prediction

logger = logging.getLogger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id        TEXT PRIMARY KEY,
    transaction_id    INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    threshold         REAL NOT NULL,
    fraud_probability REAL NOT NULL,
    prediction        INTEGER NOT NULL,
    risk_level        TEXT NOT NULL,
    features          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction ON predictions (transaction_id);
"""

INSERT = "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"


@dataclass(frozen=True)
class PredictionRecord:
    request_id: str
    transaction_id: int
    created_at: str
    model_name: str
    model_version: str
    threshold: float
    fraud_probability: float
    prediction: int
    risk_level: str
    features: dict[str, Any]


class PredictionLogger(Protocol):
    def log(self, records: list[PredictionRecord]) -> None: ...


def build_records(
    predictions: list[Prediction], inputs: list[dict[str, Any]], model_name: str
) -> list[PredictionRecord]:
    """Pair each prediction with the input that produced it."""
    now = datetime.now(UTC).isoformat()
    return [
        PredictionRecord(
            request_id=p.request_id,
            transaction_id=p.transaction_id,
            created_at=now,
            model_name=model_name,
            model_version=p.model_version,
            threshold=p.threshold,
            fraud_probability=p.fraud_probability,
            prediction=p.prediction,
            risk_level=p.risk_level,
            features=x,
        )
        for p, x in zip(predictions, inputs, strict=True)
    ]


class SQLitePredictionLogger:
    """Local implementation. Opens a short-lived connection per call (thread-safe)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript(CREATE_TABLE)

    def log(self, records: list[PredictionRecord]) -> None:
        rows = [
            (
                r.request_id,
                r.transaction_id,
                r.created_at,
                r.model_name,
                r.model_version,
                r.threshold,
                r.fraud_probability,
                r.prediction,
                r.risk_level,
                json.dumps(r.features),
            )
            for r in records
        ]
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.executemany(INSERT, rows)


def log_safely(prediction_logger: PredictionLogger, records: list[PredictionRecord]) -> None:
    """Never let a logging failure reach the client: record it in our logs instead."""
    try:
        prediction_logger.log(records)
    except Exception:
        logger.exception("Failed to log %d predictions", len(records))
