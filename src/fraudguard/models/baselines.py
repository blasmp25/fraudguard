"""Baseline pipelines: shared preprocessing -> model-specific encoding -> estimator."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from fraudguard.features.preprocessing import FeatureSpec, FraudPreprocessor

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
MISSING = "__missing__"


def build_pipeline(name: str, spec: FeatureSpec, seed: int) -> Pipeline:
    """Return an unfitted pipeline that takes raw transaction columns."""
    prep = FraudPreprocessor(spec)
    num_cols = [*spec.numeric, "hour"]
    cat_cols = list(spec.categorical)

    if name == "logistic_regression":
        encode = ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value=MISSING),
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    cat_cols,
                ),
            ]
        )
        model = LogisticRegression(max_iter=2000)
        return Pipeline([("prep", prep), ("encode", encode), ("model", model)])

    if name == "random_forest":
        encode = ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value=MISSING),
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                    cat_cols,
                ),
            ]
        )
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("prep", prep), ("encode", encode), ("model", model)])

    if name == "xgboost":
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("prep", prep), ("model", model)])

    if name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        return Pipeline([("prep", prep), ("model", model)])

    raise ValueError(f"Unknown model: {name}")
