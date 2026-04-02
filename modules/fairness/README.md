# Fairness Module

This module is designed as a practical, first-line fairness audit for tabular machine learning workflows. It focuses on interpretable, implementable metrics that surface bias, error asymmetries, and performance disparities across demographic subgroups, while acknowledging that fairness is inherently context-dependent and requires human judgment to act on.

---

## Files

```
modules/fairness/
│
├── src/
│   ├── fairness_metrics.py         ← Metric computation (EOD, DPR, accuracy diff, error rates)
│   ├── subgroup_analysis.py        ← Per-group breakdown of accuracy, TPR, FPR, FNR
│   ├── bias_detection.py           ← Threshold evaluation and flag assignment
│   ├── fairness_pipeline.py        ← Orchestrator: runs all checks, emits unified report
│   └── fairness_utils.py           ← Shared helpers (bootstrap CI, base rates, group splits)
│
├── data/
│   ├── processed/                  ← Cleaned versions of Adult Income and German Credit
│   └── subgroup_splits/            ← Pre-split data per sensitive attribute
│
├── notebooks/
│   ├── fairness_exploration.ipynb        ← EDA and subgroup distribution analysis
│   ├── fairness_metrics_validation.ipynb ← End-to-end pipeline demo (main deliverable)
│   └── subgroup_analysis.ipynb           ← Deep-dive per sensitive attribute
│
├── outputs/
│   ├── reports/                    ← JSON reports consumed by the dashboard
│   ├── plots/                      ← Grouped bar charts, metric scorecards
│   └── tables/                     ← Per-group metric summaries as CSV
│
├── tests/
│   ├── test_fairness_metrics.py
│   └── test_subgroup_logic.py
│
└── README.md                       ← this file
```

---

## Installation

All dependencies for this module are listed in the root `requirements.txt`. Before running anything, make sure your environment is set up:

1. Create and activate a virtual environment from the project root.
2. Install all dependencies using `pip install -r requirements.txt`.
3. Confirm that `fairlearn` and `scikit-learn` are installed correctly by checking their versions in your environment.
4. `aif360` is an optional dependency used only if you need in-processing mitigation algorithms. It is not required for the core metric pipeline — `fairlearn` covers all four metrics in this module.

If you encounter dependency conflicts between `fairlearn` and `aif360`, install them in separate environments and run only what you need for your task.

---

## Implementation Instructions

This section describes what you need to build and in what order. We will the fairlearn and scikit-learn documentation linked in the References section for implementation guidance.

**Step 1 — Set up the data pipeline**
Load the Adult Income and German Credit datasets from UCI. Clean each dataset: handle missing values, encode categorical columns, and separate the target column from the features. Do not touch the sensitive attribute columns beyond ensuring they are clean and consistently labeled. Create an 80/20 stratified train/test split using a fixed random seed so results are reproducible across runs.

**Step 2 — Train a baseline classifier**
Train a logistic regression classifier on the training split using only non-sensitive features. Do not include `sex`, `race`, or `personal_status` as model inputs — these are only used for evaluation, never for training. Generate predictions on the test split and attach them to your test DataFrame as a new column.

**Step 3 — Implement the metric functions in `fairness_metrics.py`**
Implement each of the four metrics as standalone functions that accept ground truth labels, predicted labels, and a sensitive feature Series as inputs. Use fairlearn's `MetricFrame` to compute per-group metrics. Refer to the formulas and thresholds defined in this README. Each function should return a dictionary containing the metric value, confidence interval, and status (PASS / WARNING / FLAG).

**Step 4 — Implement bootstrap confidence intervals in `fairness_utils.py`**
For each metric function, wrap the computation in a bootstrap loop (1,000 iterations). On each iteration, resample the test set with replacement and recompute the metric. Use the 2.5th and 97.5th percentiles of the resulting distribution as the confidence interval bounds. Return the CI half-width alongside the metric value.

**Step 5 — Implement subgroup analysis in `subgroup_analysis.py`**
Compute per-group accuracy, TPR, FPR, and FNR by splitting the test set on the sensitive feature and computing a confusion matrix for each group. These values feed directly into the JSON output schema — make sure the keys match exactly.

**Step 6 — Implement the interpretability layer in `bias_detection.py`**
Apply the thresholds from `fairness_utils.py` to each metric value and assign PASS, WARNING, or FLAG status. Generate a human-readable direction-of-harm string for each flagged or warning metric that names the disadvantaged group and describes the type of disparity.

**Step 7 — Build the pipeline orchestrator in `fairness_pipeline.py`**
Wire together steps 3–6 into a single class that runs all checks in sequence and assembles the full JSON report. The report must conform exactly to the Standard Output Schema defined in this README. Implement a `save_report()` method that writes the JSON to `reports/fairness/<timestamp>_<dataset>.json`.

**Step 8 — Write the demo notebook**
In `notebooks/fairness_metrics_validation.ipynb`, demonstrate the full pipeline end-to-end: load Adult Income → train model → generate predictions → run FairnessPipeline for `sex` → run again for `race` → save both reports → print a summary. This notebook is the primary deliverable for the Week 4 milestone.

**Step 9 — Validate against fairlearn baselines**
Cross-check your EOD and DPR values against fairlearn's built-in `equalized_odds_difference` and `demographic_parity_ratio` functions called directly on the same data. Results should match within ±0.01. Document this validation in `notebooks/fairness_metrics_validation.ipynb`.

---

Before running this module, you need to have completed three steps: loaded and cleaned your dataset, trained a classifier on the training split, and generated predictions on the test split. The pipeline takes those predictions as input — it does not train a model itself.

Once predictions are ready, instantiate the pipeline for one sensitive attribute at a time. Fairness metrics are always computed per attribute independently — do not combine attributes in a single run.

```python
from modules.fairness.src.fairness_pipeline import FairnessPipeline

pipeline = FairnessPipeline(
    df=df,                                  # test split only, with predictions attached
    target_col="income",                    # ground truth column
    prediction_col="predicted_income",      # model output column
    sensitive_feature="sex",               # one attribute per run
)

report = pipeline.run_all_metrics()
pipeline.save_report(report, dataset="adult_income")
print(report["status"])                     # PASS | FAIL | WARNING
```

Repeat the run for each sensitive attribute (e.g., call once with `"sex"`, once with `"race"`).
Each run produces a separate JSON report. Each metric can also be computed independently — see `src/fairness_metrics.py`.

---

## What Is Fairness Auditing?

A model that achieves high overall accuracy may still behave differently
across demographic subgroups. Without subgroup-level evaluation, these
disparities go undetected until they cause harm.

Fairness auditing surfaces those disparities by computing group-level
metrics alongside global ones. The goal is not to prove a model is fair —
no metric set can do that — but to make disparities visible and force
an intentional decision about them.

**Why this matters:**
A model that correctly identifies 79% of qualified male applicants
but only 65% of equally qualified female applicants is not a success.
The overall accuracy number will not reveal this.

---

## Metrics

This module computes four complementary fairness metrics.
No single metric is sufficient — all must be reported together.

---

### 1. Equal Opportunity Difference (Primary)

**What it measures:**
The difference in True Positive Rate (TPR) between the most and least
favored groups. TPR is the probability that a truly positive individual
is correctly classified as positive.

```
Equal Opportunity Difference = TPR(privileged group) − TPR(unprivileged group)
```

**Why it is primary:**
For merit-based tasks (income prediction, credit risk), the central fairness
concern is whether equally qualified individuals are treated equally.
Equal opportunity focuses exactly on that: are qualified individuals in each group
being correctly identified at the same rate?

**Threshold:**

```python
EQUAL_OPPORTUNITY_THRESHOLD = 0.10   # flag if |EOD| > 0.10
```

**Risk levels:**

| Level | Condition | Meaning |
|---|---|---|
| PASS | \|EOD\| ≤ 0.10 | Groups have similar TPR |
| WARNING | 0.10 < \|EOD\| ≤ 0.15 | Moderate disparity — investigate |
| FLAG | \|EOD\| > 0.15 | Significant disparity — act before deployment |

**Example output:**

```
Equal Opportunity Difference : 0.14 ± 0.03   ← WARNING
→ Model favors Male group in correctly identifying positives
→ Female group is under-selected despite qualification
```

**Important:**
EOD is asymmetric — it tells you the direction of harm, not just the magnitude.
Always report which group is disadvantaged, not just the absolute value.

---

### 2. Demographic Parity Ratio (Secondary)

**What it measures:**
The ratio of positive prediction rates between the unprivileged and privileged groups.
A ratio of 1.0 means both groups receive positive predictions at the same rate.

```
Demographic Parity Ratio = P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)
```

**Why it is secondary:**
Demographic parity is an important complementary signal, but it is not always
the right optimization target. When base rates differ between groups (as they do
in Adult Income), enforcing equal prediction rates is not equivalent to enforcing
equal treatment. It is reported for context, not for optimization.

**Threshold:**

```python
DEMOGRAPHIC_PARITY_RATIO_THRESHOLD = 0.80   # flag if ratio < 0.80
```

**Risk levels:**

| Level | Condition | Meaning |
|---|---|---|
| PASS | DPR ≥ 0.80 | Prediction rates broadly comparable |
| WARNING | 0.70 ≤ DPR < 0.80 | Borderline — document and monitor |
| FLAG | DPR < 0.70 | Substantial prediction rate gap |

**Example output:**

```
Demographic Parity Ratio : 0.73   ← Borderline
→ Female group receives positive predictions at 73% the rate of Male group
```

**Important:**
Interpreting DPR requires knowing the base rates. If the Male base rate is 0.42
and Female is 0.28, a DPR below 1.0 does not automatically indicate unfairness.
Always read DPR alongside the Base Rate Awareness section.

---

### 3. Accuracy Difference Across Groups (Tertiary)

**What it measures:**
The absolute difference in classification accuracy between the best-performing
and worst-performing subgroup.

```
Accuracy Difference = Accuracy(best group) − Accuracy(worst group)
```

**Why it is tertiary:**
Accuracy differences are easy to communicate to non-technical stakeholders and
provide a useful sanity check. However, accuracy can be misleading on imbalanced
datasets and does not capture which type of error is driving the gap.
It is reported for transparency, not for threshold-based decisions.

**Threshold:**

```python
ACCURACY_DIFFERENCE_THRESHOLD = 0.05   # flag if gap > 5%
```

**Example output:**

```
Accuracy Difference : 0.06   ← Investigate
  Male   : 0.87
  Female : 0.81
```

**Important:**
A 6% accuracy gap sounds small but may reflect a systematic disadvantage.
Always drill into FPR and FNR to understand what is driving it.

---

### 4. Error Rate Disparity (Diagnostic)

**What it measures:**
Two complementary error-rate gaps across groups:

- **FPR Difference** — difference in False Positive Rate (rate of incorrect positive classifications)
- **FNR Difference** — difference in False Negative Rate (rate of missed positive cases)

```
FPR Difference = FPR(unprivileged) − FPR(privileged)
FNR Difference = FNR(unprivileged) − FNR(privileged)
```

Both formulas follow the same convention: unprivileged minus privileged.
A positive value means the unprivileged group bears a higher error rate.
A negative value means the privileged group bears a higher error rate — report it as such.

**Why it is diagnostic:**
FPR and FNR reveal the type of harm being imposed. In credit and income contexts:
- High FPR for a group → over-penalization (false accusations of risk)
- High FNR for a group → under-recognition (missed qualification)

These are qualitatively different harms. Reporting them together exposes
whether the model is failing a group by punishing it more or by recognizing
it less.

**Thresholds:**

```python
FPR_DIFFERENCE_THRESHOLD = 0.10   # flag if |FPR diff| > 0.10
FNR_DIFFERENCE_THRESHOLD = 0.10   # flag if |FNR diff| > 0.10
```

**Example output:**

```
FPR Difference : 0.12   ← FLAG
→ One group experiences elevated false positive rate
→ Suggests potential over-penalization

FNR Difference : 0.09   ← Investigate
→ One group has more missed positive cases
→ Suggests under-recognition among qualified individuals
```

---

## Thresholds Summary

```python
FAIRNESS_THRESHOLDS = {
    "equal_opportunity_difference":   0.10,   # flag if |EOD| > 0.10
    "demographic_parity_ratio":       0.80,   # flag if ratio < 0.80
    "accuracy_difference":            0.05,   # flag if gap > 5%
    "fpr_difference":                 0.10,   # flag if |FPR diff| > 0.10
    "fnr_difference":                 0.10,   # flag if |FNR diff| > 0.10
}
```

Thresholds are defined in `src/fairness_utils.py` and can be overridden when
instantiating `FairnessPipeline`.

---

## Base Rate Awareness

The module reports base rates per group before any metric computation:

```
Base rates (P(Y=1)):
  Male   : 0.42
  Female : 0.28
```

**Why this matters:**
Differences in base rates explain why fairness metrics can conflict.
Demographic parity requires equal prediction rates across groups — but if one
group genuinely has more positive cases in the data, enforcing equal rates means
either over-predicting for the lower-base-rate group or under-predicting for the
higher-base-rate group. Neither is obviously fair.

Always read fairness metrics in light of base rates. A DPR of 0.73 when base
rates are 0.42 vs. 0.28 carries a different interpretation than the same DPR
when base rates are equal.

---

## Statistical Confidence

All reported fairness metrics include uncertainty estimates computed via
bootstrap resampling (1,000 iterations by default):

```
Equal Opportunity Difference : 0.14 ± 0.03
Demographic Parity Ratio     : 0.73 ± 0.04
FPR Difference               : 0.12 ± 0.02
```

**Why this matters:**
A metric value of 0.11 that just crosses the 0.10 threshold should be
interpreted differently from a value of 0.20. Confidence intervals prevent
over-reaction to borderline values and under-reaction to large ones.

If the confidence interval of a flagged metric crosses the threshold, mark the
finding as WARNING rather than FLAG and note the uncertainty in the report.

---

## Direction of Harm (Interpretability Layer)

Every metric output includes a human-readable interpretation that names the
disadvantaged group and describes the type of harm:

```
Equal Opportunity Difference : 0.14
→ Model favors Male group in correctly identifying positives
→ Female group is under-selected despite qualification

FPR Difference : 0.12
→ Elevated false positive rate for Female group
→ Suggests potential over-penalization of one group
```

This layer exists because a number alone is not actionable. Stakeholders need to
understand who is harmed and how before they can decide what to do about it.

---

## Metric Selection Rationale

Equal Opportunity Difference is the primary metric for both datasets because:

- Both tasks are merit-based (income prediction, credit risk)
- The fairness concern is whether equally qualified individuals are treated equally
- Demographic parity is not appropriate when base rates differ between groups
- EOD directly measures the gap in correct identification of qualified individuals

Other metrics are reported for context and completeness, not for optimization.
Attempting to optimize all metrics simultaneously will result in trade-offs that
make the model worse on the primary concern.

---

## Standard Output Schema

Every pipeline run emits a JSON report to `reports/fairness/<timestamp>_<dataset>.json`.
This schema must be respected — it is what the dashboard reads.

```json
{
  "module": "fairness",
  "dataset": "adult_income",
  "sensitive_feature": "sex",
  "model": "LogisticRegression",
  "timestamp": "2024-11-01T14:32:00Z",
  "status": "FAIL",
  "summary": "Significant equal opportunity gap detected: Female group correctly identified at 14pp lower rate than Male group.",
  "base_rates": {
    "Male": 0.42,
    "Female": 0.28
  },
  "metrics": {
    "equal_opportunity_difference": {"value": 0.14, "ci": 0.03, "status": "WARNING"},
    "demographic_parity_ratio":     {"value": 0.73, "ci": 0.04, "status": "WARNING"},
    "accuracy_difference":          {"value": 0.06, "ci": 0.02, "status": "FLAG"},
    "fpr_difference":               {"value": 0.12, "ci": 0.02, "status": "FLAG"},
    "fnr_difference":               {"value": 0.09, "ci": 0.03, "status": "WARNING"}
  },
  "subgroup_accuracy": {
    "Male": 0.87,
    "Female": 0.81
  },
  "subgroup_tpr": {
    "Male": 0.79,
    "Female": 0.65
  },
  "subgroup_fpr": {
    "Male": 0.14,
    "Female": 0.26
  },
  "subgroup_fnr": {
    "Male": 0.21,
    "Female": 0.35
  },
  "flags": ["accuracy_difference", "fpr_difference"],
  "recommendations": [
    "Investigate feature engineering for sex-correlated proxies.",
    "Consider post-processing threshold adjustment to equalize TPR across groups.",
    "Re-evaluate on German Credit dataset to check whether pattern generalizes."
  ]
}
```

---

## Datasets

### Adult Income Dataset (UCI ML Repository)

Primary benchmark for subgroup fairness evaluation.

- **Target:** `income` (binary: ≤50K / >50K)
- **Sensitive features:** `sex` (binary), `race` (multi-class — evaluate pairwise)
- **Known consideration:** The dataset reflects 1994 U.S. census data. The
  `sex` column is binary and does not capture gender identity. `race` categories
  are simplified encodings of complex social categories. Acknowledge these
  limitations in the final report.
- **Split:** 80/20 stratified split, fixed random seed for reproducibility

### German Credit Dataset

Secondary benchmark for fairness in financial decision systems.

- **Target:** credit risk (binary: good / bad)
- **Sensitive features:** `age`, `personal_status` (encodes gender and marital status)
- **Known consideration:** `personal_status` conflates gender and marital status.
  Separate these when possible — do not treat `personal_status` as a single
  sensitive attribute without acknowledging what it encodes.
- **Split:** 80/20 stratified split, fixed random seed

---

## Example Full Output

```
Dataset             : adult_income
Sensitive attribute : sex
Groups detected     : ['Male', 'Female']
Timestamp           : 2024-11-01T14:32:00Z

--- Base Rates ---
  Male   : 0.42
  Female : 0.28

--- Metric Results ---
Equal Opportunity Difference  :  0.14 ± 0.03   ← WARNING
Demographic Parity Ratio      :  0.73 ± 0.04   ← Borderline
Accuracy Difference           :  0.06 ± 0.02   ← FLAG
FPR Difference                :  0.12 ± 0.02   ← FLAG
FNR Difference                :  0.09 ± 0.03   ← Investigate

--- Interpretation ---
→ Model favors Male group in correctly identifying positives
→ Female group is under-selected despite qualification
→ Elevated FPR for Female group indicates potential over-penalization
→ Confidence intervals do not cross thresholds — findings are stable

--- Per-Group Results ---
  Accuracy  |  Male: 0.87   Female: 0.81
  TPR       |  Male: 0.79   Female: 0.65
  FPR       |  Male: 0.14   Female: 0.26
  FNR       |  Male: 0.21   Female: 0.35

--- Status ---
  FAIL — 2 metrics flagged, 3 at WARNING level

--- Recommendations ---
  → Investigate sex-correlated proxy features in feature set
  → Consider post-processing threshold adjustment for TPR equalization
  → Re-run audit on German Credit to assess generalizability
```

---

## Evaluation Criteria

The module is evaluated on three dimensions:

**1. Metric correctness (against known baselines)**

| Metric | Pass condition |
|---|---|
| Equal Opportunity Difference | Matches fairlearn `equalized_odds_difference` output within ±0.01 |
| Demographic Parity Ratio | Matches fairlearn `demographic_parity_ratio` within ±0.01 |
| FPR / FNR Difference | Consistent with manual per-group confusion matrix computation |

**2. Confidence interval coverage**

Bootstrap CI must be reported for all primary and secondary metrics.
CI width should be ≤ 0.05 for datasets of standard size (n > 5,000).

**3. Output schema compliance**

Every report file must parse as valid JSON and include all required fields:
`module`, `dataset`, `sensitive_feature`, `timestamp`, `status`, `summary`,
`metrics`, `flags`, `recommendations`. The dashboard will reject reports
missing these fields.

---

## Important to understand:

**Metric incompatibility.**
Fairness definitions mathematically conflict. It is impossible to simultaneously
satisfy equal opportunity, demographic parity, and calibration when base rates
differ (Chouldechova, 2017). Do not attempt to optimize for all metrics — choose
the primary metric appropriate for the task and report others for context.

**Dataset scope.**
Adult Income reflects 1994 U.S. census data. German Credit reflects a specific
European financial context. Patterns found on these datasets may not generalize.
Always re-run fairness audits on new datasets before deployment.

**No metric proves fairness.**
Metrics surface disparities. They do not explain their cause, confirm
discrimination occurred, or prescribe a remedy. Interpretation requires
domain knowledge and stakeholder judgment.

**Sensitive feature limitations.**
Both datasets encode sensitive attributes as simplified binary or categorical
variables. These encodings do not capture the full complexity of gender, race,
or socioeconomic status. Acknowledge this explicitly in any report produced
from this module.

---

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities.* MIT Press. https://fairmlbook.org/
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments.* Big Data, 5(2).
- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
- fairlearn Documentation: https://fairlearn.org/
- aif360 Documentation: https://aif360.readthedocs.io/
- UCI ML Repository — Adult Income Dataset: https://archive.ics.uci.edu/dataset/2/adult
- UCI ML Repository — German Credit Dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
