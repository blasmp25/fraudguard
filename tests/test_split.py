import pandas as pd
import pytest

from fraudguard.config.settings import SplitConfig
from fraudguard.data.split import SECONDS_PER_DAY, temporal_split


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
