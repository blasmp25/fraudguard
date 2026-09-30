from pathlib import Path

import pandas as pd

from fraudguard.data.load import IDENTITY_FILE, TRANSACTION_FILE, read_raw


def test_read_raw_left_join_sorted_and_downcast(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [300, 100, 200],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "isFraud": [0, 1, 0],
        }
    ).to_csv(tmp_path / TRANSACTION_FILE, index=False)
    pd.DataFrame({"TransactionID": [2], "id_01": [-5.0]}).to_csv(
        tmp_path / IDENTITY_FILE, index=False
    )

    df = read_raw(tmp_path)

    assert len(df) == 3  # no transaction lost by the join
    assert df["TransactionDT"].is_monotonic_increasing  # sorted by time
    assert df["id_01"].isna().sum() == 2  # only one transaction has identity
    assert df["TransactionAmt"].dtype == "float32"  # downcast applied
