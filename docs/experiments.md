# Experiments

## Phase 1 — Baselines (validation set, days 120–149)

**Setup:** temporal split (train: days < 120), 60 selected features + `hour`, shared preprocessing
(rare categories grouped, nulls kept), no resampling, no class weights, no early stopping,
500 trees for boosting models. Threshold chosen to maximise F1 on the validation set.

| Model | PR-AUC | ROC-AUC | Recall @ P=0.5 | Threshold | Precision | Recall | F1 | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.609 | 0.925 | 0.607 | 0.185 | 0.651 | 0.527 | 0.582 | 26 s |
| XGBoost | 0.606 | 0.926 | 0.605 | 0.211 | 0.692 | 0.496 | 0.578 | 45 s |
| Random Forest | 0.522 | 0.909 | 0.501 | 0.175 | 0.547 | 0.474 | 0.508 | 171 s |
| Logistic Regression | 0.399 | 0.830 | 0.342 | 0.184 | 0.486 | 0.351 | 0.408 | 51 s |

**Champion:** `fraudguard@champion` = LightGBM (version 1, threshold 0.185).

### Observations

- **PR-AUC separates the models far more than ROC-AUC** (0.21 vs 0.10 between best and worst).
  With ~82k legitimate transactions in validation, thousands of false positives barely move the
  false positive rate, so ROC-AUC looks good for every model; precision is hit directly.
- **LightGBM and XGBoost are tied** (0.609 vs 0.606, within the ±1–2% run-to-run noise observed
  during feature selection). LightGBM was chosen because it trains ~1.7× faster.
- **At the chosen threshold, 65% of alerts are real fraud but 47% of fraud is missed.**
  Accepting 50% precision would raise recall to ~61%. Improving recall is the goal of phase 3.
- **Removing early stopping and adding `hour` / rare-category grouping had no measurable effect**
  (0.609 vs 0.607–0.611 in feature selection).
- The best threshold (~0.19) is far from 0.5 because the model is trained on 3.5% prevalence,
  so predicted probabilities are low; the threshold is a business decision, here set by max F1.
