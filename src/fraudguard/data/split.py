"""Temporal train / validation / future split based on TransactionDT."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from fraudguard.config.settings import SplitConfig

SECONDS_PER_DAY = 86400


class TemporalSplit(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    future: pd.DataFrame


def transaction_day(df: pd.DataFrame) -> pd.Series:
    """Day index of each transaction (TransactionDT is in seconds)."""
    return df["TransactionDT"] // SECONDS_PER_DAY


def temporal_split(df: pd.DataFrame, cfg: SplitConfig) -> TemporalSplit:
    """Split chronologically: train < train_end_day <= valid < valid_end_day <= future."""
    day = transaction_day(df)
    train = df[day < cfg.train_end_day].copy()
    valid = df[(day >= cfg.train_end_day) & (day < cfg.valid_end_day)].copy()
    future = df[day >= cfg.valid_end_day].copy()
    return TemporalSplit(train=train, valid=valid, future=future)
