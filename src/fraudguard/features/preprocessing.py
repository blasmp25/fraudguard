"""Feature preprocessing shared by training, evaluation and serving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from fraudguard.config import PROJECT_ROOT
from fraudguard.data.split import SECONDS_PER_DAY

OTHER = "__other__"
DEFAULT_FEATURES_FILE = PROJECT_ROOT / "configs" / "features.yaml"


@dataclass(frozen=True)
class FeatureSpec:
    """Which raw columns the model uses, as listed in configs/features.yaml."""

    numeric: list[str]
    categorical: list[str]

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_FEATURES_FILE) -> FeatureSpec:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(numeric=list(raw["numeric"]), categorical=list(raw["categorical"]))


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Learn categories on train (fit) and apply the same transformation anywhere (transform)."""

    def __init__(self, spec: FeatureSpec, min_category_count: int = 50) -> None:
        self.spec = spec
        self.min_category_count = min_category_count

    @property
    def output_columns(self) -> list[str]:
        return [*self.spec.numeric, "hour", *self.spec.categorical]

    def fit(self, df: pd.DataFrame, y: object = None) -> FraudPreprocessor:
        self.categories_: dict[str, list[str]] = {}
        for col in self.spec.categorical:
            counts = df[col].value_counts()
            frequent = counts[counts >= self.min_category_count].index.astype(str)
            self.categories_[col] = sorted(frequent) + [OTHER]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "categories_"):
            raise RuntimeError("FraudPreprocessor must be fitted before transform")

        df = df.copy()
        for col in [*self.spec.numeric, *self.spec.categorical, "TransactionDT"]:
            if col not in df.columns:
                df[col] = np.nan

        out = pd.DataFrame(index=df.index)
        for col in self.spec.numeric:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        seconds = pd.to_numeric(df["TransactionDT"], errors="coerce")
        out["hour"] = ((seconds % SECONDS_PER_DAY) // 3600).astype("float32")

        for col in self.spec.categorical:
            known = self.categories_[col]
            values = df[col].astype("string")
            values = values.where(values.isna() | values.isin(known), OTHER)
            out[col] = pd.Categorical(values, categories=known)

        return out[self.output_columns]
