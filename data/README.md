# Data

This project uses the **IEEE-CIS Fraud Detection** dataset (Vesta Corporation), published on Kaggle:
https://www.kaggle.com/c/ieee-fraud-detection

The data is **not included in this repository**: the competition rules do not allow redistribution.
Each user must download it with their own Kaggle account.

## How to get it

1. Create a Kaggle account and open the competition page.
2. Go to the **Rules** tab and click **I Understand and Accept**.
3. In Kaggle, go to **Settings → API → Create New Token** and save the token:
   - Linux/macOS: `~/.kaggle/access_token`
   - Windows: `%USERPROFILE%\.kaggle\access_token`
   - Or set the `KAGGLE_API_TOKEN` environment variable.
4. From the repository root, run:

```bash
uv run python scripts/download_data.py
```

The script downloads only the labelled training files into `data/raw/` and skips files that already exist.

## Files

| File | Rows | Description |
| --- | --- | --- |
| `raw/train_transaction.csv` | ~590k | One row per transaction; target column `isFraud` |
| `raw/train_identity.csv` | ~144k | Device and network information, joined on `TransactionID` (not every transaction has it) |

The `test_*` files from the competition are not used: they have no labels.

## Notes

- `TransactionDT` is a time offset in seconds from an undisclosed reference date, not a real timestamp.
  It covers about six months and is used for the temporal train / validation / "future production" split.
- The data is highly imbalanced (~3.5 % fraud).
- Most feature names are anonymised (`C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339`).
