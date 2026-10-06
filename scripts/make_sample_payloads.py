"""Write example /predict payloads (one fraud, one legit) from the validation set.

Usage (from the repository root):
    uv run python scripts/make_sample_payloads.py
"""

import json

import pandas as pd

from fraudguard.config import load_settings
from fraudguard.data.load import load_dataset
from fraudguard.data.split import temporal_split
from fraudguard.features.preprocessing import FeatureSpec


def to_payload(row: pd.Series, spec: FeatureSpec) -> dict[str, object]:
    """Convert one dataset row into the JSON body expected by /predict."""
    payload: dict[str, object] = {
        "transaction_id": int(row["TransactionID"]),
        "TransactionDT": int(row["TransactionDT"]),
    }
    for col in [*spec.numeric, *spec.categorical]:
        value = row[col]
        if pd.isna(value):
            payload[col] = None
        elif hasattr(value, "item"):  # numpy scalar -> plain Python number
            payload[col] = value.item()
        else:
            payload[col] = value
    return payload


def main() -> None:
    settings = load_settings()
    spec = FeatureSpec.from_yaml()
    valid = temporal_split(load_dataset(settings), settings.split).valid

    out_dir = settings.data.raw_dir.parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for label, name in [(1, "fraud"), (0, "legit")]:
        row = valid[valid["isFraud"] == label].iloc[0]
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(to_payload(row, spec), indent=2), encoding="utf-8")
        print(f"Written {path} (transaction {int(row['TransactionID'])})")


if __name__ == "__main__":
    main()
