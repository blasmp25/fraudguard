import json
import sqlite3
from pathlib import Path

from fraudguard.api.prediction_log import PredictionRecord, SQLitePredictionLogger


def make_record(request_id: str) -> PredictionRecord:
    return PredictionRecord(
        request_id=request_id,
        transaction_id=1,
        created_at="2026-10-07T10:00:00+00:00",
        model_name="fraudguard",
        model_version="1",
        threshold=0.185,
        fraud_probability=0.9,
        prediction=1,
        risk_level="HIGH",
        features={"TransactionAmt": 50.0, "P_emaildomain": None},
    )


def test_records_are_persisted_with_features(tmp_path: Path) -> None:
    db = tmp_path / "predictions.db"
    SQLitePredictionLogger(db).log([make_record("a"), make_record("b")])

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT request_id, features FROM predictions").fetchall()
    assert [r[0] for r in rows] == ["a", "b"]
    assert json.loads(rows[0][1]) == {"TransactionAmt": 50.0, "P_emaildomain": None}


def test_restarting_does_not_lose_history(tmp_path: Path) -> None:
    db = tmp_path / "predictions.db"
    SQLitePredictionLogger(db).log([make_record("a")])
    SQLitePredictionLogger(db).log([make_record("b")])  # simulates an API restart

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2
