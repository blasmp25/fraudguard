from pathlib import Path

import pytest
from pydantic import ValidationError

from fraudguard.config import PROJECT_ROOT, load_settings


def test_local_config_loads() -> None:
    settings = load_settings()
    assert settings.split.train_end_day < settings.split.valid_end_day
    assert settings.data.raw_dir == PROJECT_ROOT / "data" / "raw"


def test_invalid_split_order_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data: {raw_dir: data/raw, processed_dir: data/processed}\n"
        "split: {train_end_day: 150, valid_end_day: 120}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_settings(bad)
