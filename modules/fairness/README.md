# Leakage Detection Module

This module helps you catch data leakage before it silently inflates model's performance. It runs four checks on your dataset and flags features or patterns that look suspicious. It won't catch every possible form of leakage and it may occasionally flag something that turns out to be fine — but it gives you a structured starting point and forces a documented decision on anything it finds.

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

## Installation

All dependencies for this module are in the root `requirements.txt`. Here's how to get set up:

1. Create and activate a virtual environment from the project root.
2. Run `pip install -r requirements.txt` to install everything.
3. Double-check that `scikit-learn`, `pandas`, and `scipy` installed correctly by checking their versions.
4. `great-expectations` is optional. It's useful if you want to add formal data contracts on top of the checks this module already does, but it has a lot of setup overhead. You don't need it to run the core pipeline — `pandas` and `scipy` handle everything.

---

## Implementation Instructions

This section walks you through what to build and in what order. There's no code here — use the scikit-learn, pandas, and scipy documentation linked in the References section to figure out the specifics as you go.

**Step 1 — Load and prepare the data**
Download the Adult Income dataset from UCI. Clean it: handle missing values, encode categorical columns, and make sure the target column is separated from the features. Create an 80/20 stratified train/test split with a fixed random seed. Keep both splits as separate DataFrames — the pipeline takes them as distinct inputs. Don't shuffle or merge them after splitting.

**Step 2 — Build the synthetic dataset first**
Before you write any check logic, build the synthetic dataset in `data/synthetic/`. You need three versions: one where a column is basically a noisy copy of the target label, one where 5% of test rows are exact duplicates from the training set, and one where a date column contains values that go past the reference date. Build this dataset first because you'll use it to test each check as you build it — it's much easier to debug that way.

**Step 3 — Build Check 1 in `target_correlation.py`**
For every feature column, measure how strongly it's correlated with the target. Use Pearson correlation for numeric features and Spearman correlation for categorical ones (after encoding). Assign each feature a severity level based on the threshold bands in the Thresholds Reference section and return the raw correlation values alongside the severity for the report.

**Step 4 — Build Check 2 in `train_test_contamination.py`**
Hash every row in the train and test DataFrames (all feature columns, not the target). Count how many row hashes appear in both sets, compute the contamination ratio as `duplicate_rows / len(df_test)`, and assign a severity level. Return the count, ratio, and severity.

**Step 5 — Build Check 3 in `future_information.py`**
Take a list of datetime column names and a `reference_date`. For each column, check whether any values go past that date and flag the ones that do. Make both parameters optional — if the user doesn't provide them, the check should report itself as SKIPPED, not LOW or PASS.

**Step 6 — Build Check 4 in `single_feature_audit.py`**
For each feature, train a logistic regression using only that one feature on the training split, then check its ROC-AUC on the test split. Assign a severity level based on the AUC threshold bands. To keep this from taking too long, only run this check on features that already came back MEDIUM or HIGH from Check 1, plus a small random sample of the rest. Note this decision in your notebook.

**Step 7 — Set up shared helpers in `leakage_utils.py`**
Put the threshold constants in one place so every check reads from the same source. Add helper functions for row hashing, correlation, and any formatting you reuse across checks. The `LEAKAGE_THRESHOLDS` dictionary from this README should live here and be imported by each check module — don't hardcode values inside the individual files.

**Step 8 — Build the pipeline in `leakage_pipeline.py`**
Create a class that takes the train DataFrame, test DataFrame, target column name, and optional temporal parameters. Run all four checks in sequence, collect their outputs, compute `overall_risk` using the aggregation logic from the Risk Levels section, and assemble the full JSON report. Add a `save_report()` method that writes to `reports/leakage/<timestamp>_<dataset>.json`. The report must match the Standard Output Schema in this README exactly.

**Step 9 — Write the demo notebook**
In `notebooks/leakage_demo.ipynb`, show the full flow: load Adult Income → split the data → run `LeakagePipeline` → save the report → print a readable summary of what was flagged. This is your main Week 4 deliverable.

**Step 10 — Validate with the synthetic dataset**
In `notebooks/synthetic_leakage_audit.ipynb`, run the pipeline against all three synthetic scenarios. Each check must return HIGH for the leakage it was built to catch. If one doesn't fire, the problem is in the implementation logic — go back and fix it before touching the threshold values. Write up the validation results as a table in the notebook.

---

## How to Use

Before running this module, you need two things ready: the dataset loaded and split into separate train and test DataFrames, and the target column identified and confirmed as binary. Unlike the fairness module, you do not need a trained model or predictions — the leakage pipeline operates directly on the raw data and the split itself.

If you want to run Check 3, you also need to identify which columns in your dataset are datetime columns and confirm your `reference_date` — the moment at which a prediction would actually be made in production. Do not guess this value; confirm it with whoever defined the data pipeline.

```python
from modules.leakage.src.leakage_pipeline import LeakagePipeline

pipeline = LeakagePipeline(
    df_train=train_df,
    df_test=test_df,
    target_col="income",
    datetime_cols=["transaction_date"],     # optional — omit if no temporal columns
    reference_date="2023-01-01",            # optional — required only for Check 3
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

Primary benchmark for leakage checks on real-world tabular data. Predicts whether an individual earns more than $50,000/year. Attributes include age, education, occupation, marital status, race, and gender.

- **Target:** `income` (binary: ≤50K / >50K)
- **Sensitive features:** `sex`, `race`
- **Known risk:** `fnlwgt` (census sampling weight) may produce elevated correlations
  — investigate before treating as leakage; it is a sampling artifact, not a target proxy
- **Split:** 80/20 stratified split; fix the random seed for reproducibility

### Synthetic Leakage Dataset

Generated to simulate controlled leakage scenarios for testing and validating the leakage detection module. Three scenarios are implemented:

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

### What this module doesn't catch

This module covers four common leakage patterns. There are others it won't detect:

- **Aggregate leakage:** if you compute group-level statistics (like mean income by zip code) across the whole dataset before splitting, that information leaks into the test set — and this module won't catch it
- **Pipeline leakage:** if you fit a scaler or encoder on training + test data combined, the test set influenced the preprocessing — also not caught here
- **Proxy leakage:** a feature that isn't the target but is derived from it through a less obvious path — hard to detect automatically
- **Evaluation design issues:** for example, cross-validation folds with overlapping time windows

These are real problems worth knowing about, even if they're out of scope here.

### The thresholds are judgment calls

A correlation of 0.96 might be direct leakage or a genuinely strong predictor — the module can't tell. A correlation of 0.60 might be a subtle proxy leak below the detection threshold. The checks will sometimes flag things that are fine (false positives) and sometimes miss things that aren't (false negatives). Always review flagged features with knowledge of where they came from.

### Your inputs have to be right

The module is only as good as what you give it. A few things that will cause misleading results even if the code is correct:

- If `target_col` points to a downstream label rather than the actual prediction target, Check 1 and Check 4 will be measuring the wrong thing
- If you leave datetime columns out of `datetime_cols`, Check 3 won't see them
- If `reference_date` doesn't reflect the true moment of prediction, Check 3 will produce incorrect results — this one is easy to get wrong, so confirm the date carefully

---

## Summary

This module gives you a structured, documented first pass at leakage detection. It's designed to be practical for a graduate project — interpretable checks, clear outputs, and a validation path against synthetic data. It won't catch everything, and every flag it raises still needs a human to look at it and decide what to do. Use it as a starting point, not a final answer.

---

## References

- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM Transactions on Knowledge Discovery from Data, 6(4).
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the Reproducibility Crisis in Machine Learning-based Science.* Patterns, 4(9).
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE, 10(3).
- UCI ML Repository — Adult Income Dataset: https://archive.ics.uci.edu/dataset/2/adult
- scikit-learn — `roc_auc_score`, `train_test_split`: https://scikit-learn.org/stable/
- Great Expectations — Data quality and validation framework: https://docs.greatexpectations.io/
