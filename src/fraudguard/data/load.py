"""Load the raw IEEE-CIS files, merge them and store a single Parquet file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudguard.config import Settings

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
PROCESSED_FILE = "train.parquet"


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 columns to float32 to halve memory usage."""
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def read_raw(raw_dir: Path) -> pd.DataFrame:
    """Read both raw CSVs, left-join identity onto transactions, sort by time."""
    transactions = pd.read_csv(raw_dir / TRANSACTION_FILE)
    identity = pd.read_csv(raw_dir / IDENTITY_FILE)
    df = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return _downcast_floats(df)


def build_processed(settings: Settings) -> Path:
    """Create data/processed/train.parquet from the raw CSVs."""
    df = read_raw(settings.data.raw_dir)
    settings.data.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.data.processed_dir / PROCESSED_FILE
    df.to_parquet(out, index=False)
    return out


def load_dataset(settings: Settings) -> pd.DataFrame:
    """Load the processed dataset. Run scripts/make_dataset.py first."""
    path = settings.data.processed_dir / PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/make_dataset.py")
    return pd.read_parquet(path)
