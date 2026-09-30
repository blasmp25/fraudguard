"""Merge the raw Kaggle files into data/processed/train.parquet.

Usage:
    uv run python scripts/make_dataset.py
"""

import time

from fraudguard.config import load_settings
from fraudguard.data.load import build_processed, load_dataset


def main() -> None:
    settings = load_settings()

    start = time.perf_counter()
    out = build_processed(settings)
    print(f"Written {out} in {time.perf_counter() - start:.0f}s")

    start = time.perf_counter()
    df = load_dataset(settings)
    elapsed = time.perf_counter() - start
    memory_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"Reloaded from Parquet in {elapsed:.1f}s")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns, {memory_gb:.2f} GB in memory")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")


if __name__ == "__main__":
    main()
