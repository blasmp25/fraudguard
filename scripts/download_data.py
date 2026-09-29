"""Download the IEEE-CIS Fraud Detection training data from Kaggle.

Usage:
    uv run python scripts/download_data.py

Requires a Kaggle API token (see data/README.md) and having accepted
the competition rules on kaggle.com.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "ieee-fraud-detection"
FILES = ["train_transaction.csv", "train_identity.csv"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def extract_zips(folder: Path) -> None:
    """Unzip every .zip in folder and delete the archive."""
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    for filename in FILES:
        if (RAW_DIR / filename).exists():
            print(f"[skip] {filename} already downloaded")
            continue
        print(f"[download] {filename}")
        api.competition_download_file(COMPETITION, filename, path=RAW_DIR, quiet=False)
        extract_zips(RAW_DIR)

    print(f"Done. Files in {RAW_DIR}")


if __name__ == "__main__":
    main()
