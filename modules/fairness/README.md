# Fairness Module

Checks whether a trained classifier treats demographic groups fairly. Computes fairness metrics across subgroups, flags disparities above defined thresholds, and produces a JSON report consumed by the dashboard.

This module does not prove fairness — that always requires human judgment. It surfaces the numbers you need to have that conversation.

---

## Files

```
modules/fairness/
│
├── src/
│   ├── fairness_metrics.py         ← EOD and accuracy-gap computation + thresholds
│   ├── subgroup_analysis.py        ← Per-group accuracy, TPR, FPR, FNR
│   ├── fairness_pipeline.py        ← Orchestrator: runs checks, emits JSON report
│   └── fairness_utils.py           ← Threshold constants and shared helpers
│
├── data/
│   └── processed/                  ← Cleaned Adult Income dataset
│
├── notebooks/
│   └── fairness_demo.ipynb         ← End-to-end demo (main deliverable)
│
├── outputs/
│   └── reports/                    ← JSON reports consumed by the dashboard
│
├── tests/
│   └── test_fairness_metrics.py    ← 2 tests (Student D, Week 5)
│
└── README.md                       ← this file
```

---

## Installation

All dependencies are in the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required: `pandas`, `numpy`, `scikit-learn`, `fairlearn`. No `aif360` — fairlearn alone covers every metric in this module.

---

## Implementation Instructions

Build in this order. Each step uses only what was built in earlier steps.

**Step 1 — Load and prepare Adult Income.**
Download Adult Income from UCI. Handle missing values, encode categorical columns, separate the target (`income`) from features. Keep the `sex` column clean for evaluation but do not use it as a model input. Create an 80/20 stratified train/test split with a fixed random seed for reproducibility.

**Step 2 — Train a baseline model.**
Train a logistic regression on the training split using non-sensitive features only. Never include `sex` or `race` as model inputs — those columns exist only for fairness evaluation. Generate predictions on the test split and attach them to the test DataFrame as a new column. The pipeline takes this DataFrame as input — it does not train a model itself.

**Step 3 — Build the two metric functions in `fairness_metrics.py`.**
One function for Equal Opportunity Difference, one for Accuracy Difference. Each takes ground truth labels, predicted labels, and the sensitive-feature column, and returns the metric value and a status (PASS / WARNING / FLAG) using the thresholds from `fairness_utils.py`. Use fairlearn's `MetricFrame` — it handles per-group splitting for you.

**Step 4 — Build per-group breakdowns in `subgroup_analysis.py`.**
Split the test set by the sensitive feature, compute a confusion matrix per group, and extract accuracy, TPR, FPR, and FNR per group. These values go directly into the JSON report — make sure the key names match the schema exactly.

**Step 5 — Wire it together in `fairness_pipeline.py`.**
Build a `FairnessPipeline` class that runs steps 3 and 4 in sequence and assembles the JSON report matching the Standard Output Schema below. Add a `save_report()` method that writes to `outputs/reports/<timestamp>_<dataset>.json`.

**Step 6 — Write the demo notebook.**
In `notebooks/fairness_demo.ipynb`: load Adult Income → train model → generate predictions → run the pipeline for `sex` → save the report → print a readable summary. This is the main deliverable.

**Step 7 — Sanity-check against fairlearn.**
Call fairlearn's built-in `equalized_odds_difference` on the same data and confirm your EOD matches within ±0.01. If it doesn't, there's a bug in your metric logic — fix it before moving on. Document the check in the notebook.

---

## How to Use

Before running: the Adult Income dataset is cleaned and split, a classifier is trained, and predictions are attached to the test DataFrame.

```python
from modules.fairness.src.fairness_pipeline import FairnessPipeline

pipeline = FairnessPipeline(
    df=df_test,                             # test split with predictions attached
    target_col="income",                    # ground truth column
    prediction_col="predicted_income",      # model output column
    sensitive_feature="sex",                # one attribute per run
)

report = pipeline.run_all_metrics()
pipeline.save_report(report, dataset="adult_income")
print(report["status"])                     # PASS | FAIL | WARNING
```

Fairness metrics are computed per sensitive attribute independently. Run the pipeline once per attribute (once for `sex`, once for `race` if you want to evaluate both).

---

## What Is Fairness Auditing?

A model can have great overall accuracy while performing much worse for specific groups of people. If you only look at the overall number, you miss this completely.

Fairness auditing breaks down model performance by demographic group so the gaps become visible. The goal isn't to prove the model is fair — no set of metrics can do that — but to make sure you know where the disparities are before the model gets used.

A model that correctly identifies 79% of qualified male applicants but only 65% of equally qualified female applicants is not a good model — and the overall accuracy number will never show you that.

---

## Metrics

This module computes two fairness metrics. Both are reported together.

### 1. Equal Opportunity Difference (Primary)

**What it measures:** The gap in True Positive Rate (TPR) between groups. TPR is the share of people who actually qualify for a positive outcome that the model correctly identifies. A large gap means the model is much better at spotting qualified people in one group than another.

```
EOD = TPR(group A) − TPR(group B)
```

**Why it's primary:** For merit-based tasks like income prediction, the core fairness question is whether equally qualified people are treated equally. EOD answers this directly.

**Thresholds:**

| Level | Condition | Meaning |
|---|---|---|
| PASS | \|EOD\| ≤ 0.10 | Groups have similar TPR |
| WARNING | 0.10 < \|EOD\| ≤ 0.15 | Moderate gap — worth investigating |
| FLAG | \|EOD\| > 0.15 | Large gap — needs attention before deployment |

**Example output:**

```
Equal Opportunity Difference : 0.14   ← WARNING
→ Female group under-identified by 14 percentage points
```

The sign of EOD tells you which group is being disadvantaged, not just the magnitude. Always report which group is affected, not just the number.

### 2. Accuracy Difference (Secondary)

**What it measures:** The absolute difference in classification accuracy between the best-performing and worst-performing subgroup.

```
Accuracy Difference = Accuracy(best group) − Accuracy(worst group)
```

**Why it's secondary:** Accuracy differences are easy to communicate but don't reveal *which type* of error is driving the gap. Reported alongside EOD for context and as a sanity check.

**Threshold:** Flag if gap > 0.05.

**Example output:**

```
Accuracy Difference : 0.06   ← FLAG
  Male   : 0.87
  Female : 0.81
```

---

## Thresholds

```python
FAIRNESS_THRESHOLDS = {
    "equal_opportunity_difference": 0.10,   # flag if |EOD| > 0.10
    "accuracy_difference":          0.05,   # flag if gap > 0.05
}
```

Thresholds live in `src/fairness_utils.py` and can be overridden when instantiating `FairnessPipeline`.

---

## Base Rate Awareness

Before computing any metric, the pipeline reports base rates per group:

```
Base rates (P(Y=1)):
  Male   : 0.42
  Female : 0.28
```

Base-rate differences explain why fairness metrics can conflict with each other. If one group genuinely has more positive cases in the data, forcing equal prediction rates means over-predicting for the lower-base-rate group or under-predicting for the higher-base-rate group — neither is obviously fair.

Always read fairness metrics in light of base rates.

---

## Metric Selection Rationale

Equal Opportunity Difference is the primary metric because:
- The task is merit-based (income prediction)
- The fairness concern is whether equally qualified individuals are treated equally
- EOD directly measures the gap in correct identification of qualified individuals
- Demographic parity is not appropriate here because base rates differ between groups

Accuracy Difference is reported for transparency and as a sanity check, not for optimization.

---

## Standard Output Schema

Every pipeline run emits a JSON report to `outputs/reports/<timestamp>_<dataset>.json`. The dashboard reads this schema — do not change field names.

```json
{
  "module": "fairness",
  "dataset": "adult_income",
  "sensitive_feature": "sex",
  "model": "LogisticRegression",
  "timestamp": "2026-04-17T14:32:00Z",
  "status": "FAIL",
  "summary": "Equal opportunity gap detected: Female group correctly identified at 14pp lower rate than Male group.",
  "base_rates": {
    "Male": 0.42,
    "Female": 0.28
  },
  "metrics": {
    "equal_opportunity_difference": {"value": 0.14, "status": "WARNING"},
    "accuracy_difference":          {"value": 0.06, "status": "FLAG"}
  },
  "subgroup_accuracy": {"Male": 0.87, "Female": 0.81},
  "subgroup_tpr":      {"Male": 0.79, "Female": 0.65},
  "subgroup_fpr":      {"Male": 0.14, "Female": 0.26},
  "subgroup_fnr":      {"Male": 0.21, "Female": 0.35},
  "flags": ["accuracy_difference"]
}
```

**Status mapping:**
- Any metric flagged → `status = "FAIL"`
- Any metric at WARNING → `status = "WARNING"`
- Otherwise → `status = "PASS"`

---

## Dataset

**Adult Income (UCI ML Repository).** Predicts whether an individual earns more than $50,000/year. Target: `income` (binary, ≤50K / >50K). Sensitive features: `sex` (binary), `race` (multi-class — evaluate pairwise).

Known limitations:
- 1994 U.S. census data — findings do not generalize to other populations
- Binary `sex` encoding does not capture gender identity
- `race` categories are simplified encodings of complex social categories

Acknowledge these limitations in the final write-up. Split: 80/20 stratified with a fixed random seed.

---

## Example Full Output

```
Dataset             : adult_income
Sensitive attribute : sex
Groups detected     : ['Male', 'Female']
Timestamp           : 2026-04-17T14:32:00Z

--- Base Rates ---
  Male   : 0.42
  Female : 0.28

--- Metric Results ---
Equal Opportunity Difference  : 0.14   ← WARNING
Accuracy Difference           : 0.06   ← FLAG

--- Per-Group Results ---
  Accuracy  |  Male: 0.87   Female: 0.81
  TPR       |  Male: 0.79   Female: 0.65
  FPR       |  Male: 0.14   Female: 0.26
  FNR       |  Male: 0.21   Female: 0.35

--- Status ---
  FAIL — 1 metric flagged, 1 at WARNING
```

---

## Evaluation Criteria

The module is evaluated on two dimensions.

**1. Metric correctness.** EOD matches fairlearn's `equalized_odds_difference` within ±0.01 on the same data. This is enforced by a test in `tests/test_fairness_metrics.py`.

**2. Output schema compliance.** Every JSON report parses cleanly and includes every required field from the Standard Output Schema. The dashboard rejects reports missing any required field.

---

## Things to Keep in Mind

**You can't optimize all fairness metrics at once.** Chouldechova (2017) showed that equal opportunity, demographic parity, and calibration mathematically conflict when groups have different base rates. This module picks EOD as the primary metric for that reason — don't try to force other metrics to pass alongside it.

**The dataset is old and narrow.** Adult Income reflects 1994 U.S. census conditions. Don't generalize findings to other populations or use cases.

**A passing score doesn't prove fairness.** These metrics show where disparities exist. They can't tell you *why* the disparity is there, whether discrimination occurred, or what to do about it. That part requires domain understanding and stakeholder conversation.

**The sensitive attributes are simplified.** Binary `sex` and simplified `race` categories don't capture real identity. Be honest about this limitation in write-ups.

---

## References

- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact.* Big Data, 5(2).
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning.* MIT Press. https://fairmlbook.org/
- fairlearn documentation: https://fairlearn.org/
- UCI Adult Income dataset: https://archive.ics.uci.edu/dataset/2/adult
