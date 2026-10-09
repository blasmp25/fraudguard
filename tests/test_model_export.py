from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.api.model_service import ModelService, export_model, load_exported
from fraudguard.features.preprocessing import FeatureSpec
from fraudguard.models.baselines import build_pipeline

SPEC = FeatureSpec(numeric=["TransactionAmt", "C1"], categorical=["P_emaildomain"])


def test_export_roundtrip_predicts_identically(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "TransactionDT": rng.integers(86_400, 200 * 86_400, 200),
            "TransactionAmt": rng.lognormal(4, 1, 200),
            "C1": rng.integers(0, 10, 200).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], 200),
        }
    )
    y = (df["TransactionAmt"] > 100).astype(int)
    pipeline = build_pipeline("logistic_regression", SPEC, seed=0).fit(df, y)
    original = ModelService(
        model=pipeline, spec=SPEC, name="fraudguard", alias="champion", version="3", threshold=0.2
    )

    export_model(original, tmp_path)
    loaded = load_exported(tmp_path)

    np.testing.assert_array_equal(loaded.model.predict_proba(df), original.model.predict_proba(df))
    assert (loaded.version, loaded.threshold) == ("3", 0.2)
    assert loaded.spec == SPEC
