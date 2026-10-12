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

## Phase 3 — Imbalance-handling study: protocol

**Status:** frozen on 2026-10-12, before running any Phase 3 experiment. Changes after this date are listed in an amendments section with their reason.

### Research question

Which imbalance-handling strategy gives the best ranking of fraudulent transactions (PR-AUC) under a realistic temporal evaluation, and at what cost in probability calibration?

### Design

| Factor | Levels |
| --- | --- |
| Strategy | `none`, `class_weight`, `random_undersampling`, `smote`, `borderline_smote`, `adasyn` |
| Model | LightGBM, Logistic Regression |
| Validation | 3 expanding-window temporal folds, 7-day gap between train and validation |
| Seeds | 0, 1, 2, 3, 4 (used by both the resampler and the model) |

6 strategies × 2 models × 3 folds × 5 seeds = **180 runs**, executed as a SLURM job array on the Urania cluster (UCA).

**Planned addendum:** CorSMOTE (the author's method, unpublished) will be added later under exactly the same folds, seeds, representation, ratio and hyperparameters; the statistical analysis will then be repeated with all strategies.

### Folds (days < 150 only; the future set, days ≥ 150, is not used)

| Fold | Train | Gap | Validation |
| --- | --- | --- | --- |
| 1 | days 1–52 | 53–59 | 60–89 |
| 2 | days 1–82 | 83–89 | 90–119 |
| 3 | days 1–112 | 113–119 | 120–149 |

The gap mimics label delay (chargebacks arrive days or weeks later). Fold 1's training data is dominated by the holiday period, which EDA showed to be atypical (low fraud rate); results on fold 1 are interpreted with that in mind.

### Representation (identical for every strategy and model)

Fitted on the training part of each fold only, inside the pipeline:

1. Shared preprocessing (`FraudPreprocessor`: 60 selected features + `hour`, rare categories grouped).
2. Numeric features: median imputation + missing-value indicators + standardisation.
3. Categorical features: frequency encoding (share of each category in the training fold).

This is required by the neighbour-based methods (SMOTE family), and applying it everywhere ensures that **only the resampling step differs between strategies**.

### Strategies

- **Resampling ratio:** minority / majority = 0.5 after resampling (`sampling_strategy=0.5`) for every resampler. SMOTE-family methods use `k_neighbors=5`.
- **`class_weight`:** positive-class weight chosen so that the effective ratio is also 1:2 (weight = 0.5 × n_negative / n_positive).
- Resampling is applied **only to the training part** of each fold, inside an `imblearn.Pipeline`; validation data is never resampled.

### Models (fixed hyperparameters, not tuned per strategy)

- LightGBM: same hyperparameters as the Phase 1 baseline (500 trees, learning rate 0.05, 64 leaves, subsample 0.8, colsample 0.5).
- Logistic Regression: `max_iter=2000`, default regularisation.

### Metrics (on each validation fold)

| Metric | Role |
| --- | --- |
| PR-AUC | **Primary** |
| Recall at precision ≥ 0.5 | Secondary (fixed business operating point, no threshold selection) |
| ROC-AUC | Reported for reference |
| Brier score (raw probabilities) | Calibration cost of each strategy |
| F1, G-Mean at best-F1 threshold | Reported as **oracle** (optimistic) values, for comparison between strategies only |
| Training time | Practical cost |

### Statistical analysis

- For each model separately: Friedman test over the 15 blocks (3 folds × 5 seeds), α = 0.05.
- If significant: Nemenyi post-hoc test and a critical-difference diagram.
- Results reported as mean ± standard deviation over the 15 blocks.

### Reference (outside the statistical comparison)

The production LightGBM (native NaN handling, no resampling, no frequency encoding) is evaluated on the same folds, to know whether any strategy beats the current champion setup.

### Final step

The best strategy (by mean PR-AUC, if supported by the statistical test) is retrained on days 1–142 and evaluated **once** on the future set (days ≥ 150), alongside the current champion. It is promoted to `@champion` only if its future PR-AUC is higher and the 95% bootstrap confidence interval of the difference excludes zero.

### Known limitations

- A single resampling ratio (0.5) is studied; its effect is not explored.
- Hyperparameters are not tuned per strategy; a strategy could benefit from different settings.
- Frequency encoding loses category identity when two categories share a frequency.
