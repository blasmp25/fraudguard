import pandas as pd
import pytest
from pydantic import ValidationError

from fraudguard.config.settings import CvConfig, Settings, SplitConfig
from fraudguard.data.split import SECONDS_PER_DAY, temporal_folds, temporal_split


@pytest.fixture
def df() -> pd.DataFrame:
    """One transaction per day, days 0..199."""
    days = range(200)
    return pd.DataFrame(
        {
            "TransactionID": list(days),
            "TransactionDT": [d * SECONDS_PER_DAY + 10 for d in days],
            "isFraud": [d % 2 for d in days],
        }
    )


CFG = SplitConfig(train_end_day=120, valid_end_day=150)


def test_no_rows_lost_or_duplicated(df: pd.DataFrame) -> None:
    split = temporal_split(df, CFG)
    assert len(split.train) + len(split.valid) + len(split.future) == len(df)


def test_sets_are_strictly_chronological(df: pd.DataFrame) -> None:
    split = temporal_split(df, CFG)
    assert split.train["TransactionDT"].max() < split.valid["TransactionDT"].min()
    assert split.valid["TransactionDT"].max() < split.future["TransactionDT"].min()


def test_no_transaction_in_two_sets(df: pd.DataFrame) -> None:
    split = temporal_split(df, CFG)
    ids = [set(s["TransactionID"]) for s in split]
    assert ids[0].isdisjoint(ids[1])
    assert ids[1].isdisjoint(ids[2])
    assert ids[0].isdisjoint(ids[2])


def test_boundary_day_goes_to_later_set(df: pd.DataFrame) -> None:
    split = temporal_split(df, CFG)
    assert 120 in set(split.valid["TransactionID"])
    assert 150 in set(split.future["TransactionID"])


CV = CvConfig(valid_starts=[60, 90, 120], valid_days=30, gap_days=7)


def days_of(part: pd.DataFrame) -> pd.Series:
    return part["TransactionDT"] // SECONDS_PER_DAY


def test_folds_match_protocol(df: pd.DataFrame) -> None:
    folds = temporal_folds(df, CV)
    assert [
        (days_of(f.train).max(), days_of(f.valid).min(), days_of(f.valid).max()) for f in folds
    ] == [(52, 60, 89), (82, 90, 119), (112, 120, 149)]


def test_gap_separates_train_and_valid(df: pd.DataFrame) -> None:
    for f in temporal_folds(df, CV):
        assert days_of(f.valid).min() - days_of(f.train).max() > CV.gap_days


def test_training_windows_expand(df: pd.DataFrame) -> None:
    folds = temporal_folds(df, CV)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert set(earlier.train["TransactionID"]) < set(later.train["TransactionID"])


def test_folds_never_use_future_set(df: pd.DataFrame) -> None:
    for f in temporal_folds(df, CV):
        assert days_of(f.valid).max() < 150


def test_config_rejects_folds_reaching_future() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
                "split": {"train_end_day": 120, "valid_end_day": 150},
                "cv": {"valid_starts": [60, 90, 130], "valid_days": 30, "gap_days": 7},
            }
        )
