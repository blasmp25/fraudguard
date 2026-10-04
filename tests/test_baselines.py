import numpy as np
import pandas as pd
import pytest

from fraudguard.features.preprocessing import FeatureSpec
from fraudguard.models.baselines import MODEL_NAMES, build_pipeline

SPEC = FeatureSpec(numeric=["TransactionAmt", "C1"], categorical=["P_emaildomain"])


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "TransactionDT": rng.integers(86_400, 200 * 86_400, n),
            "TransactionAmt": rng.lognormal(4, 1, n),
            "C1": rng.integers(0, 10, n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        }
    )
    y = pd.Series((df["TransactionAmt"] > 100).astype(int))
    return df, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_outputs_valid_probabilities(
    name: str, data: tuple[pd.DataFrame, pd.Series]
) -> None:
    df, y = data
    pipeline = build_pipeline(name, SPEC, seed=0).fit(df, y)
    proba = pipeline.predict_proba(df)[:, 1]
    assert proba.shape == (len(df),)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        build_pipeline("svm", SPEC, seed=0)
