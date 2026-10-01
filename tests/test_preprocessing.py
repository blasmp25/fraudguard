import numpy as np
import pandas as pd
import pytest

from fraudguard.features.preprocessing import OTHER, FeatureSpec, FraudPreprocessor

SPEC = FeatureSpec(numeric=["TransactionAmt"], categorical=["P_emaildomain"])


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [86_400 + 3 * 3600, 86_400, 90_000, 100_000, 110_000],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "rare.com", None],
        }
    )


@pytest.fixture
def prep(train: pd.DataFrame) -> FraudPreprocessor:
    return FraudPreprocessor(SPEC, min_category_count=2).fit(train)


def test_rare_categories_grouped_in_other(prep: FraudPreprocessor) -> None:
    # gmail appears twice (kept); yahoo and rare appear once (grouped)
    assert prep.categories_["P_emaildomain"] == ["gmail.com", OTHER]


def test_unseen_category_maps_to_other(prep: FraudPreprocessor) -> None:
    out = prep.transform(pd.DataFrame({"P_emaildomain": ["new-domain.com"]}))
    assert out["P_emaildomain"].iloc[0] == OTHER


def test_missing_stays_missing_not_other(prep: FraudPreprocessor) -> None:
    out = prep.transform(pd.DataFrame({"P_emaildomain": [None]}))
    assert pd.isna(out["P_emaildomain"].iloc[0])


def test_hour_of_day(prep: FraudPreprocessor, train: pd.DataFrame) -> None:
    out = prep.transform(train)
    assert out["hour"].iloc[0] == 3


def test_missing_columns_filled_with_nan(prep: FraudPreprocessor) -> None:
    out = prep.transform(pd.DataFrame({"TransactionAmt": [5.0]}))
    assert list(out.columns) == prep.output_columns
    assert np.isnan(out["hour"].iloc[0])


def test_single_row_matches_batch(prep: FraudPreprocessor, train: pd.DataFrame) -> None:
    """Serving one transaction must give exactly what batch processing gives."""
    batch = prep.transform(train)
    single = prep.transform(train.iloc[[2]])
    pd.testing.assert_frame_equal(single, batch.iloc[[2]])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        FraudPreprocessor(SPEC).transform(pd.DataFrame({"TransactionAmt": [1.0]}))
