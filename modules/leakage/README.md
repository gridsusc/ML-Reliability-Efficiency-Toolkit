# Leakage Detection Module

This module is designed as a practical, first-line leakage audit for tabular machine learning workflows. It focuses on interpretable, implementable checks that surface suspicious patterns associated with leakage, while acknowledging that leakage detection is inherently context-dependent and requires human validation.

---

## Files

```
modules/leakage/
│
├── src/
│   ├── target_correlation.py         ← Check 1: features with strong target association
│   ├── train_test_contamination.py   ← Check 2: duplicate rows across splits
│   ├── future_information.py         ← Check 3: temporal features postdating prediction time
│   ├── single_feature_audit.py       ← Check 4: single-feature AUC suspiciously high
│   ├── leakage_pipeline.py           ← Orchestrator: runs all checks, emits unified report
│   └── leakage_utils.py              ← Shared helpers (correlation, AUC, hashing, thresholds)
│
├── data/
│   ├── processed/                    ← Cleaned versions of Adult Income dataset
│   └── synthetic/                    ← Controlled leakage scenarios for validation
│
├── notebooks/
│   ├── leakage_exploration.ipynb     ← EDA and leakage signal discovery
│   ├── leakage_demo.ipynb            ← End-to-end pipeline demo (main deliverable)
│   └── synthetic_leakage_audit.ipynb ← Functional validation against known-leakage dataset
│
├── outputs/
│   ├── reports/                      ← JSON reports consumed by the dashboard
│   ├── plots/                        ← Correlation heatmaps, AUC bar charts
│   └── tables/                       ← Flagged feature summaries as CSV
│
├── tests/
│   ├── test_target_correlation.py
│   ├── test_contamination.py
│   └── test_single_feature_auc.py
│
└── README.md                         ← this file
```

---

## How to Use

```python
from modules.leakage.src.leakage_pipeline import LeakagePipeline

pipeline = LeakagePipeline(
    df_train=train_df,
    df_test=test_df,
    target_col="income",
    datetime_cols=["transaction_date"],     # optional
    reference_date="2023-01-01",            # optional, for Check 3
)

report = pipeline.run_all_checks()
pipeline.save_report(report, dataset="adult_income")
print(report["overall_risk"])               # HIGH | MEDIUM | LOW
```

Each check can also be run independently — see the individual `src/` files.
All thresholds can be overridden by passing a custom `thresholds` dict to `LeakagePipeline`.

---

## What Is Data Leakage?

Data leakage occurs when information that would not be legitimately available at
prediction time is used during model training. Kaufman et al. (2012) define it as
the use of information in the model training process which would not be expected
to be available at prediction time. The result is a model that appears to perform
well in evaluation but fails in production.

Leakage is not always obvious. It can enter the pipeline through:

- A feature that directly encodes the target (e.g., `approved_flag` predicts `approved`)
- Train and test data that share rows (contamination)
- A timestamp that reflects information only known after the prediction event
- A feature that, by itself, nearly perfectly separates the target classes

As Kapoor and Narayanan (2023) document, leakage is often subtle and tied to
workflow and evaluation design — not just feature-level anomalies. Detection
therefore requires both statistical checks and human judgment.

**Why this matters:**
A model with 99% accuracy trained on leaked data is not a success — it is a
silent failure waiting to be deployed.

---

## Detection Methods

This module runs four complementary checks. No single check is sufficient.
All four must be run and reported together.

---

### Check 1 — Target Correlation

**What it detects:**
Features with unusually strong statistical association to the target, which may
indicate direct target encoding, proxy leakage, or a legitimately strong predictor
requiring manual review. This is the most common entry point for leakage in
tabular datasets.

**How it works:**
- For continuous features: compute Pearson correlation with the target
- For categorical features: compute Spearman rank correlation after encoding
- Assign a severity band based on the magnitude of |r|

**Severity bands:**

| Severity | Condition | Meaning |
|---|---|---|
| LOW | \|r\| < 0.85 | No strong association signal |
| MEDIUM | 0.85 ≤ \|r\| < 0.95 | Elevated association — review feature origin |
| HIGH | \|r\| ≥ 0.95 | Very strong association — likely requires investigation |

**Why this is only a heuristic:**
Correlation measures statistical strength, not causal structure. A feature like
`credit_score` may legitimately reach r = 0.90 with a credit risk target without
any leakage. Whether a strong correlation represents leakage depends on *when*
the feature becomes available relative to the prediction event — a question that
cannot be answered by correlation alone.

**Recommended next steps when flagged:**
- Inspect the feature's origin: when was it created, and by what process?
- Verify whether the feature is derived *after* the target event (e.g., a post-approval flag)
- Run an ablation test: retrain the model without the flagged feature and compare performance
- Document explicitly whether the feature is available at prediction time

**Example output:**

```
Target Correlation Check:
  income_encoded     : r = 0.99   → HIGH
  capital_gain       : r = 0.87   → MEDIUM
  education_num      : r = 0.34   → LOW
```

---

### Check 2 — Train/Test Contamination

**What it detects:**
Exact row duplicates shared between the training set and the test set.
When the model has seen test rows during training, evaluation metrics are
inflated and do not reflect true generalization.

**How it works:**
- Hash each row in train and test (all columns, or a specified key subset)
- Count overlapping hashes
- Compute the contamination ratio: `duplicate_rows / len(df_test)`
- Assign a severity band based on the ratio

**Severity bands:**

| Severity | Condition | Meaning |
|---|---|---|
| LOW | contamination ratio = 0% | No detected duplicates |
| MEDIUM | 0% < ratio ≤ 1% | Small overlap — review split logic |
| HIGH | ratio > 1% | Substantial overlap — evaluation results are unreliable |

**Why this is only a heuristic:**
Full-row hashing catches identical rows but not near-duplicates (same entity,
slightly different values) or entity-level contamination (same person appearing
in both train and test with different rows). A 0% result does not guarantee a
clean split — it guarantees no identical rows.

**Recommended next steps when flagged:**
- Verify whether duplicates come from expected repeated entities, data versioning
  errors, or actual cross-split leakage
- If entities can appear multiple times (e.g., one customer with multiple loans),
  rebuild the split using entity-level keys rather than row-level sampling
- Re-evaluate model performance after rebuilding the split
- Document the deduplication policy in the experiment log

**Example output:**

```
Train/Test Contamination Check:
  Test rows        : 8,141
  Duplicate rows   : 97
  Contamination    : 1.19%   → HIGH
```

---

### Check 3 — Future Information Detection

**What it detects:**
Temporal features that encode information from after the prediction event.
This form of leakage is common in financial and healthcare pipelines where
event timestamps are appended during data collection — after outcomes are known —
rather than at the moment of prediction.

**How it works:**
- Accept a list of known datetime columns and a `reference_date`
- For each datetime column, identify values that postdate `reference_date`
- Assign a severity band based on the fraction and pattern of future-dated values

**Severity bands:**

| Severity | Condition | Meaning |
|---|---|---|
| LOW | No future-dated values | No temporal inconsistency detected |
| MEDIUM | Small or isolated fraction of future-dated values, or reference date uncertain | Investigate date definitions and collection timing |
| HIGH | Systematic future-dated values clearly beyond the prediction point | Temporal leakage likely present |

**Why this is only a heuristic:**
This check is entirely dependent on `reference_date` being set correctly.
If `reference_date` does not reflect the true moment of prediction, the check
may produce false positives (legitimate dates flagged) or false negatives
(actual future dates missed). Time zone differences and logging delays can
also produce apparent future dates that do not represent real leakage.

**Recommended next steps when flagged:**
- Confirm the definition of the prediction timestamp with the team that built
  the data pipeline — do not assume
- Review the full data collection timeline to understand when each datetime
  column is populated
- Remove or temporally shift any column not available at decision time
- If the reference date is uncertain, run the check with multiple candidate dates
  and document the sensitivity

**Example output:**

```
Future Information Check:
  reference_date        : 2023-01-01
  transaction_date      : max = 2023-06-15 (systematic)   → HIGH
  account_open_date     : max = 2020-11-30                → LOW
```

---

### Check 4 — Single-Feature AUC Audit

**What it detects:**
A feature that, when used as the sole predictor in a minimal classifier,
achieves very high discriminative power. This is a strong warning signal that
the feature may encode target information too directly — but interpretation
depends on the domain and the dataset's class balance.

**How it works:**
- For each feature, train a logistic regression using only that feature
- Compute ROC-AUC on held-out data
- Assign a severity band based on the AUC value

**Severity bands:**

| Severity | Condition | Meaning |
|---|---|---|
| LOW | AUC < 0.80 | No strong single-feature signal |
| MEDIUM | 0.80 ≤ AUC < 0.95 | Notable single-feature power — review feature provenance |
| HIGH | AUC ≥ 0.95 | Very strong signal — investigate before training |

**Why this is only a heuristic:**
A very high single-feature AUC is unusual in real-world tabular data and often
signals a problem, but it may also reflect a legitimately dominant predictor in
certain domains. On highly imbalanced datasets, ROC-AUC can understate or obscure
the severity of the signal; precision-recall analysis may be more informative in
those cases (Saito & Rehmsmeier, 2015). PR-AUC is noted here as a future extension.

**Recommended next steps when flagged:**
- Inspect the semantic meaning of the feature and its relationship to the outcome
- Verify whether the feature is a transformed or encoded version of the target
- Run an ablation test: retrain the full model without the flagged feature
  and compare held-out performance — a large drop confirms dependence
- Document the finding even if the feature is ultimately retained

**Example output:**

```
Single-Feature AUC Audit:
  target_encoded_col : AUC = 0.99   → HIGH
  capital_gain       : AUC = 0.81   → MEDIUM
  age                : AUC = 0.62   → LOW
```

---

## Risk Levels

Each check produces a severity level (LOW / MEDIUM / HIGH) per feature or column.
The pipeline aggregates these into a single overall risk rating for the dataset.

| Level | Meaning | Required action |
|---|---|---|
| **HIGH** | Strong statistical signal consistent with leakage | Investigate before training; document decision to retain or remove |
| **MEDIUM** | Elevated signal — could be leakage or a legitimate strong predictor | Review manually; record the rationale |
| **LOW** | No strong leakage signal detected | Proceed, noting that LOW does not guarantee absence of leakage |

**Overall risk aggregation logic:**

```python
if any check returns HIGH:
    overall_risk = "HIGH"
elif any check returns MEDIUM:
    overall_risk = "MEDIUM"
else:
    overall_risk = "LOW"
```

---

## Thresholds Reference

All thresholds are heuristic. They are defined in `src/leakage_utils.py`
and can be overridden at instantiation time.

```python
LEAKAGE_THRESHOLDS = {
    # Check 1 — Target Correlation
    "target_correlation_medium": 0.85,    # |r| >= this → MEDIUM
    "target_correlation_high":   0.95,    # |r| >= this → HIGH

    # Check 2 — Train/Test Contamination
    "duplicate_row_ratio_medium": 0.0,    # any duplicate → MEDIUM
    "duplicate_row_ratio_high":   0.01,   # ratio > 1% → HIGH

    # Check 3 — Future Information
    # MEDIUM: any future-dated value or uncertain reference date
    # HIGH: clear systematic future-dated values

    # Check 4 — Single-Feature AUC
    "single_feature_auc_medium": 0.80,    # AUC >= this → MEDIUM
    "single_feature_auc_high":   0.95,    # AUC >= this → HIGH
}
```

---

## Standard Output Schema

Every pipeline run emits a JSON report to `reports/leakage/<timestamp>_<dataset>.json`.
This schema is required for dashboard integration — do not omit any field.

```json
{
  "module": "leakage",
  "dataset": "adult_income",
  "timestamp": "2024-11-01T14:32:00Z",
  "status": "FAIL",
  "summary": "HIGH risk detected: target_encoded_col flagged by correlation and AUC checks.",
  "checks": {
    "target_correlation": {
      "flagged_features": ["target_encoded_col"],
      "max_correlation": 0.99,
      "risk": "HIGH"
    },
    "train_test_contamination": {
      "duplicate_rows": 0,
      "contamination_ratio": 0.0,
      "risk": "LOW"
    },
    "future_information": {
      "flagged_columns": [],
      "risk": "LOW"
    },
    "single_feature_auc": {
      "flagged_features": ["target_encoded_col"],
      "max_auc": 0.99,
      "risk": "HIGH"
    }
  },
  "overall_risk": "HIGH",
  "recommendations": [
    "Inspect provenance of target_encoded_col — flagged by both correlation and AUC checks.",
    "Run ablation test: retrain without this feature and compare held-out performance.",
    "Re-run leakage pipeline after removal to confirm overall risk drops to LOW."
  ],
  "recommended_action": [
    "Inspect feature provenance",
    "Run ablation without flagged feature",
    "Re-run pipeline after remediation"
  ]
}
```

**Status mapping:**

| `overall_risk` | `status` |
|---|---|
| HIGH | FAIL |
| MEDIUM | WARNING |
| LOW | PASS |

---

## Datasets

### Adult Income Dataset (UCI ML Repository)

Primary benchmark for leakage checks on real-world tabular data.

- **Target:** `income` (binary: ≤50K / >50K)
- **Sensitive features:** `sex`, `race`
- **Known risk:** `fnlwgt` (census sampling weight) may produce elevated correlations
  — investigate before treating as leakage; it is a sampling artifact, not a target proxy
- **Split:** 80/20 stratified split; fix the random seed for reproducibility

### Synthetic Leakage Dataset

Purpose-built to validate that each check responds correctly to known leakage patterns.

| Scenario | Injection | Check(s) expected to fire | Success criterion |
|---|---|---|---|
| Direct encoding | Feature set equal to label ± small noise | Check 1 and Check 4 | Both return HIGH |
| Train/test duplication | 5% of test rows copied from train | Check 2 | Contamination ratio ≥ 0.05, returns HIGH |
| Future timestamp | Date column includes systematic post-prediction dates | Check 3 | Column flagged as HIGH |

If a check does not fire on its corresponding synthetic scenario, the implementation
is incorrect — not the threshold.

Synthetic validation does not prove generalization to all datasets, but it does
verify that the implemented logic responds correctly to controlled leakage patterns.
The goal is not to claim the tool catches all leakage, but to confirm it catches
the leakage it was specifically designed to detect.

---

## Example Full Output

```
Dataset            : adult_income
Timestamp          : 2024-11-01T14:32:00Z

--- Check 1: Target Correlation ---
  target_encoded_col : r = 0.99   → HIGH
  capital_gain       : r = 0.87   → MEDIUM
  education_num      : r = 0.34   → LOW

--- Check 2: Train/Test Contamination ---
  Test rows          : 8,141
  Duplicate rows     : 0
  Contamination      : 0.00%     → LOW

--- Check 3: Future Information ---
  No datetime columns provided.  → SKIPPED

--- Check 4: Single-Feature AUC ---
  target_encoded_col : AUC = 0.99   → HIGH
  capital_gain       : AUC = 0.81   → MEDIUM
  age                : AUC = 0.62   → LOW

--- Overall Risk ---
  HIGH

--- Recommendations ---
  → Inspect provenance of target_encoded_col (flagged by two independent checks).
  → Run ablation: retrain without target_encoded_col and compare performance.
  → Re-run pipeline after remediation to confirm risk drops to LOW.
```

---

## Evaluation Criteria

The module is evaluated on three dimensions.

### 1. Functional Validation (Against Synthetic Dataset)

Each check must fire when the corresponding leakage is injected:

| Scenario | Injection amount | Expected check result | Pass condition |
|---|---|---|---|
| Direct encoding | Feature ≈ label | Check 1 HIGH, Check 4 HIGH | Both checks return HIGH |
| Duplication | 5% of test rows | Check 2 HIGH | Contamination ratio ≥ 0.05 |
| Future timestamp | Systematic future dates | Check 3 HIGH | Column flagged as HIGH |

### 2. False-Positive Discipline

Running the pipeline on the clean Adult Income dataset (no injected leakage,
standard 80/20 split, no engineered leakage features) must produce:

```
overall_risk = LOW
```

Excessive false positives on clean data reduce trust in the tool and make
it harder to act on legitimate flags. This criterion is as important as
detection accuracy.

### 3. Output Schema Compliance

Every report must parse as valid JSON and include all required fields:
`module`, `dataset`, `timestamp`, `status`, `summary`, `checks`,
`overall_risk`, `recommendations`. The dashboard will reject any report
that is missing these fields or uses different key names.

---

## Limitations

### A. Scope Limitations

This module targets four common, detectable forms of tabular leakage. It does
not address:

- **Aggregate leakage:** group-level statistics (e.g., mean income by zip code)
  computed over the full dataset before splitting
- **Pipeline leakage:** preprocessing steps fitted on train + test combined
  (e.g., a scaler or encoder that saw test data during fitting)
- **Proxy leakage:** features that are not the target but are derived from it
  indirectly through a non-obvious path
- **Evaluation design leakage:** e.g., cross-validation folds with overlapping
  temporal windows

These forms of leakage exist and matter. The module does not claim to detect them.

### B. Statistical Limitations

All thresholds are heuristic. A feature with r = 0.96 could be direct target
encoding or a legitimate strong predictor. A feature with r = 0.60 could be
a subtle proxy leak that this module would not flag.

The module may:
- **Over-flag** legitimate strong predictors (false positives)
- **Under-flag** subtle proxy features below threshold (false negatives)

Results must always be interpreted alongside domain knowledge and feature lineage.

### C. Operational Limitations

The module's accuracy depends on correct inputs. Specifically:

- `target_col` must be the actual prediction target, not a downstream label
- `datetime_cols` must include all temporally sensitive columns
- `reference_date` must reflect the true moment of prediction, not the data
  collection cutoff or the end of the training window
- The train/test split provided must be the actual one used for model evaluation

A misconfigured `reference_date` can cause Check 3 to produce systematically
misleading results even when the code is entirely correct.

---

## Summary

This module prioritizes interpretable first-line checks over exhaustive leakage
detection. It may miss subtle leakage introduced through preprocessing, aggregation,
feature engineering lineage, or evaluation design. Conversely, it may flag legitimate
strong predictors. Results should be interpreted as audit signals requiring human
review — not as definitive judgments about the presence or absence of leakage.

---

## References

- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM Transactions on Knowledge Discovery from Data, 6(4).
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the Reproducibility Crisis in Machine Learning-based Science.* Patterns, 4(9).
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE, 10(3).
- UCI ML Repository — Adult Income Dataset: https://archive.ics.uci.edu/dataset/2/adult
- scikit-learn — `roc_auc_score`, `train_test_split`: https://scikit-learn.org/stable/
- Great Expectations — Data quality and validation framework: https://docs.greatexpectations.io/
