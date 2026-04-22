# Leakage Detection Module

Catches common forms of data leakage before they silently inflate your model's performance. Runs three checks on a dataset and flags features or patterns that look suspicious. It won't catch every possible form of leakage, and it may occasionally flag something legitimate — but it provides a structured starting point and forces a documented decision on anything it finds.

---

## Files

```
modules/leakage/
│
├── src/
│   ├── target_correlation.py         ← Check 1: features with strong target association
│   ├── train_test_contamination.py   ← Check 2: duplicate rows across splits
│   ├── future_information.py         ← Check 3: temporal features postdating prediction time
│   ├── leakage_pipeline.py           ← Orchestrator: runs all checks, emits JSON report
│   └── leakage_utils.py              ← Threshold constants and shared helpers
│
├── data/
│   ├── processed/                    ← Cleaned Adult Income dataset
│   └── synthetic/                    ← Controlled leakage scenarios for validation
│
├── notebooks/
│   └── leakage_demo.ipynb            ← End-to-end demo + synthetic validation (main deliverable)
│
├── outputs/
│   └── reports/                      ← JSON reports consumed by the dashboard
│
├── tests/
│   └── test_leakage_checks.py        ← 3 tests (Student D, Week 5)
│
└── README.md                         ← this file
```

---

## Installation

All dependencies are in the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required: `pandas`, `numpy`, `scikit-learn`, `scipy`. No `great-expectations` — unnecessary setup overhead for the checks here.

---

## Implementation Instructions

Build in this order. Each step uses only what was built in earlier steps.

**Step 1 — Load and prepare Adult Income.**
Download from UCI. Clean it (handle missing values, encode categoricals, separate the target). Create an 80/20 stratified split with a fixed random seed. Keep train and test as separate DataFrames — the pipeline takes them as distinct inputs. Do not merge them again.

**Step 2 — Build the synthetic validation dataset first.**
Before writing any check logic, build the synthetic dataset in `data/synthetic/`. You need two scenarios:
- **Direct encoding:** a column that is essentially a noisy copy of the target label.
- **Contamination:** 5% of test rows duplicated from the training set.

Build this first because you'll use it to debug each check as you write it — it's much easier than debugging against clean data where nothing should fire.

**Step 3 — Build Check 1 in `target_correlation.py`.**
For each feature, measure correlation with the target. Pearson for numeric features, Spearman for encoded categorical features. Assign each feature a severity level (LOW / MEDIUM / HIGH) based on the thresholds in `leakage_utils.py`. Return the correlation value and severity per feature.

**Step 4 — Build Check 2 in `train_test_contamination.py`.**
Hash every row in train and test DataFrames (all feature columns, not the target). Count overlapping hashes, compute the contamination ratio as `duplicate_rows / len(df_test)`, and assign a severity level. Return the count, ratio, and severity.

**Step 5 — Build Check 3 in `future_information.py`.**
Accept a list of datetime column names and a `reference_date`. For each datetime column, check whether any values postdate the reference date. Both parameters should be optional — if not provided, the check reports as SKIPPED, not LOW or PASS.

**Step 6 — Shared helpers in `leakage_utils.py`.**
Put the threshold constants in one place so every check reads from the same source. The `LEAKAGE_THRESHOLDS` dictionary lives here and is imported by each check module — do not hardcode values inside the individual check files.

**Step 7 — Build the pipeline in `leakage_pipeline.py`.**
Create a `LeakagePipeline` class that takes the train DataFrame, test DataFrame, target column, and optional temporal parameters. Run all three checks in sequence, compute `overall_risk` using the aggregation rule in the Risk Levels section, and assemble the JSON report matching the Standard Output Schema. Add a `save_report()` method that writes to `outputs/reports/<timestamp>_<dataset>.json`.

**Step 8 — Demo and validation in `leakage_demo.ipynb`.**
Three sections in one notebook: (1) run the pipeline on clean Adult Income and verify `overall_risk = LOW`, (2) run the pipeline on the direct-encoding synthetic scenario and verify Check 1 returns HIGH, (3) run the pipeline on the contamination scenario and verify Check 2 returns HIGH. This is the main deliverable and serves as the module's functional validation.

---

## How to Use

Before running the pipeline: the dataset is split into separate train and test DataFrames, and the target column is identified and confirmed as binary. Unlike the fairness module, you don't need a trained model or predictions — the leakage pipeline operates directly on the raw data and the split itself.

If you want to run Check 3, you also need to identify which columns are datetime columns and confirm your `reference_date` — the moment at which a prediction would actually be made in production. Do not guess this value; confirm it with whoever defined the data pipeline.

```python
from modules.leakage.src.leakage_pipeline import LeakagePipeline

pipeline = LeakagePipeline(
    df_train=train_df,
    df_test=test_df,
    target_col="income",
    datetime_cols=["transaction_date"],     # optional — omit if no temporal columns
    reference_date="2023-01-01",            # required only when datetime_cols is provided
)

report = pipeline.run_all_checks()
pipeline.save_report(report, dataset="adult_income")
print(report["overall_risk"])               # HIGH | MEDIUM | LOW
```

Each check can also be run independently — see the individual `src/` files.

---

## What Is Data Leakage?

Data leakage happens when information that wouldn't be legitimately available at prediction time is used during training. Kaufman et al. (2012) define it as using information during training that the model wouldn't have when making real predictions. The result is a model that performs well in evaluation but fails in production.

Leakage enters pipelines through several common paths:
- A feature that directly encodes the target (e.g., `approved_flag` predicts `approved`)
- Train and test data sharing rows (contamination)
- Timestamps reflecting information only known after the prediction event

As Kapoor and Narayanan (2023) document, leakage is often subtle and tied to workflow and evaluation design — not just feature-level anomalies. Detection therefore requires both statistical checks and human judgment.

A model with 99% accuracy trained on leaked data is not a success — it is a silent failure waiting to be deployed.

---

## Detection Methods

This module runs three checks. All three are run and reported together.

### Check 1 — Target Correlation

**What it detects:** Features with unusually strong statistical association to the target — often a sign of direct target encoding or proxy leakage.

**How it works:** Pearson correlation for numeric features, Spearman rank correlation for encoded categorical features. Severity assigned based on the magnitude of |r|.

| Severity | Condition |
|---|---|
| LOW | \|r\| < 0.85 |
| MEDIUM | 0.85 ≤ \|r\| < 0.95 |
| HIGH | \|r\| ≥ 0.95 |

**Limitation:** Correlation measures statistical strength, not causal structure. A feature like `credit_score` may legitimately reach r = 0.90 with a credit risk target without any leakage. Whether a strong correlation represents leakage depends on *when* the feature becomes available relative to the prediction event — which this check cannot know.

**When flagged, do this:**
- Inspect the feature's origin: when was it created, by what process?
- Verify whether it's derived *after* the target event (e.g., a post-approval flag)
- Run an ablation test: retrain without the feature and compare performance

**Example output:**

```
Target Correlation Check:
  income_encoded : r = 0.99   → HIGH
  capital_gain   : r = 0.87   → MEDIUM
  education_num  : r = 0.34   → LOW
```

### Check 2 — Train/Test Contamination

**What it detects:** Exact row duplicates shared between the training set and the test set. When the model has already seen test rows during training, evaluation metrics are inflated and don't reflect true generalization.

**How it works:** Hash each row in train and test (all feature columns, not the target). Count overlapping hashes. Compute `duplicate_rows / len(df_test)`. Assign severity based on the ratio.

| Severity | Condition |
|---|---|
| LOW | ratio = 0% |
| MEDIUM | 0% < ratio ≤ 1% |
| HIGH | ratio > 1% |

**Limitation:** Full-row hashing catches identical rows but not near-duplicates (same entity, slightly different values) or entity-level contamination (same person appearing in both splits with different rows). A 0% result guarantees no identical rows — not a clean split.

**When flagged, do this:**
- Verify whether duplicates come from expected repeated entities or real leakage
- If entities can appear multiple times, rebuild the split using entity-level keys
- Re-evaluate model performance after rebuilding the split

**Example output:**

```
Train/Test Contamination Check:
  Test rows      : 8,141
  Duplicate rows : 97
  Contamination  : 1.19%   → HIGH
```

### Check 3 — Future Information Detection

**What it detects:** Temporal features that encode information from *after* the prediction event. Common in financial and healthcare pipelines where event timestamps are appended during data collection — after outcomes are known — rather than at the moment of prediction.

**How it works:** Accept a list of datetime columns and a `reference_date`. For each column, flag any values that postdate the reference. Severity based on fraction and pattern of future-dated values.

| Severity | Condition |
|---|---|
| LOW | No future-dated values |
| MEDIUM | Small or isolated fraction of future-dated values |
| HIGH | Systematic future-dated values clearly beyond the prediction point |

**Limitation:** This check is entirely dependent on `reference_date` being set correctly. If the reference date doesn't reflect the true moment of prediction, the check produces false positives or false negatives. Time zone differences and logging delays can also create apparent future dates that aren't real leakage.

**When flagged, do this:**
- Confirm the definition of the prediction timestamp with the team that built the data pipeline — do not assume
- Remove or temporally shift any column not available at decision time

**Example output:**

```
Future Information Check:
  reference_date    : 2023-01-01
  transaction_date  : max = 2023-06-15 (systematic)   → HIGH
  account_open_date : max = 2020-11-30                → LOW
```

---

## Risk Levels

Each check produces a severity level (LOW / MEDIUM / HIGH). The pipeline aggregates these into a single `overall_risk`:

```python
if any check returns HIGH:
    overall_risk = "HIGH"
elif any check returns MEDIUM:
    overall_risk = "MEDIUM"
else:
    overall_risk = "LOW"
```

| Level | Meaning | Required action |
|---|---|---|
| **HIGH** | Strong statistical signal consistent with leakage | Investigate before training; document decision to retain or remove |
| **MEDIUM** | Elevated signal — could be leakage or a legitimate strong predictor | Review manually; record the rationale |
| **LOW** | No strong leakage signal detected | Proceed, noting that LOW does not guarantee absence of leakage |

---

## Thresholds

All thresholds are heuristic. They live in `src/leakage_utils.py` and can be overridden at pipeline instantiation time.

```python
LEAKAGE_THRESHOLDS = {
    # Check 1 — Target Correlation
    "target_correlation_medium":  0.85,    # |r| >= this → MEDIUM
    "target_correlation_high":    0.95,    # |r| >= this → HIGH

    # Check 2 — Train/Test Contamination
    "duplicate_row_ratio_medium": 0.0,     # any duplicate → MEDIUM
    "duplicate_row_ratio_high":   0.01,    # ratio > 1% → HIGH

    # Check 3 — Future Information
    # MEDIUM: any future-dated value or uncertain reference date
    # HIGH:   clear systematic future-dated values
}
```

---

## Standard Output Schema

Every pipeline run emits a JSON report to `outputs/reports/<timestamp>_<dataset>.json`. The dashboard reads this schema — do not change field names.

```json
{
  "module": "leakage",
  "dataset": "adult_income",
  "timestamp": "2026-04-17T14:32:00Z",
  "status": "FAIL",
  "summary": "HIGH risk detected: target_encoded_col flagged by target-correlation check.",
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
    }
  },
  "overall_risk": "HIGH"
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

### Adult Income (UCI ML Repository)

Primary benchmark for leakage checks on real-world tabular data. Predicts whether an individual earns more than $50,000/year.

- **Target:** `income` (binary: ≤50K / >50K)
- **Known consideration:** `fnlwgt` (census sampling weight) may produce elevated correlations. Investigate before treating as leakage — it's a sampling artifact, not a target proxy.
- **Split:** 80/20 stratified with a fixed random seed.

Running the full pipeline on clean Adult Income must return `overall_risk = LOW` — this is part of the evaluation criteria.

### Synthetic Leakage Dataset

Two scenarios built specifically to validate that each check fires correctly on known leakage:

| Scenario | Injection | Expected result |
|---|---|---|
| Direct encoding | Feature ≈ noisy target label | Check 1 returns HIGH |
| Contamination | 5% of test rows copied from train | Check 2 returns HIGH, ratio ≥ 0.05 |

If a check doesn't fire on its synthetic scenario, the implementation is incorrect — fix the logic before touching thresholds.

---

## Example Full Output

```
Dataset   : adult_income
Timestamp : 2026-04-17T14:32:00Z

--- Check 1: Target Correlation ---
  target_encoded_col : r = 0.99   → HIGH
  capital_gain       : r = 0.87   → MEDIUM
  education_num      : r = 0.34   → LOW

--- Check 2: Train/Test Contamination ---
  Test rows     : 8,141
  Duplicate rows: 0
  Contamination : 0.00%   → LOW

--- Check 3: Future Information ---
  No datetime columns provided.   → SKIPPED

--- Overall Risk ---
  HIGH
```

---

## Evaluation Criteria

Three things are checked:

**1. Functional validation against the synthetic dataset.** Each check fires at HIGH when the corresponding leakage is injected. Tests in `tests/test_leakage_checks.py` enforce this.

**2. False-positive discipline on clean data.** Running on clean Adult Income produces `overall_risk = LOW`. Excessive false positives on clean data reduce trust in the tool. This criterion is as important as detection accuracy, and it is also enforced by a test.

**3. Output schema compliance.** Every JSON report parses cleanly and includes every required field from the Standard Output Schema. The dashboard rejects reports missing any required field.

---

## Limitations

**What this module doesn't catch:**
- **Aggregate leakage** — group-level statistics computed across the whole dataset before splitting
- **Pipeline leakage** — scalers or encoders fit on train + test combined
- **Proxy leakage** — features derived from the target through less obvious paths
- **Entity-level contamination** — the same person in both splits with different rows

These are real problems, but they're out of scope for this module.

**The thresholds are judgment calls.** A correlation of 0.96 might be direct leakage or a genuinely strong predictor — the module can't tell. A correlation of 0.60 might be a subtle proxy leak below the detection threshold. Always review flagged features with knowledge of where they came from.

**Garbage in, garbage out.** The module is only as good as what you give it:
- If `target_col` points to the wrong column, Check 1 measures the wrong thing
- If datetime columns are omitted from `datetime_cols`, Check 3 won't see them
- If `reference_date` doesn't reflect the true moment of prediction, Check 3 produces incorrect results — this one is easy to get wrong, so confirm carefully

---

## References

- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM Transactions on Knowledge Discovery from Data, 6(4).
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the Reproducibility Crisis in Machine Learning-based Science.* Patterns, 4(9).
- scikit-learn documentation: https://scikit-learn.org/stable/
- UCI Adult Income dataset: https://archive.ics.uci.edu/dataset/2/adult
