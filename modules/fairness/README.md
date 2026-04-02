# Fairness Module

This module helps you check whether a machine learning model treats different groups of people fairly. It computes fairness metrics across demographic subgroups and flags disparities that are large enough to investigate. It won't tell you definitively that a model is fair or unfair — that always requires human judgment — but it will surface the numbers you need to have that conversation.

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

All dependencies for this module are in the root `requirements.txt`. Here's how to get set up:

1. Create and activate a virtual environment from the project root.
2. Run `pip install -r requirements.txt` to install everything.
3. Double-check that `fairlearn` and `scikit-learn` installed correctly by checking their versions.
4. `aif360` is optional — you only need it if you want to experiment with mitigation algorithms later. The four fairness metrics in this module all run on `fairlearn` alone.

If you hit dependency conflicts between `fairlearn` and `aif360`, the easiest fix is to install them in separate environments and use only what you need.

---

## Implementation Instructions

This section walks you through what to build and in what order. There's no code here — use the fairlearn and scikit-learn documentation linked in the References section to figure out the specifics as you go.

**Step 1 — Load and prepare the data**
Download the Adult Income and German Credit datasets from UCI. For each one: handle missing values, encode any categorical columns that need it, and make sure the target column is separate from the features. Keep the sensitive attribute columns (like `sex` and `race`) clean and consistently labeled — you'll need them for evaluation, but don't use them as model inputs. Create an 80/20 stratified train/test split with a fixed random seed so your results are reproducible.

**Step 2 — Train a baseline model**
Train a logistic regression on the training split using the non-sensitive features only. Never include `sex`, `race`, or `personal_status` as inputs to the model — those columns exist only for fairness evaluation. Once trained, generate predictions on the test split and add them to your test DataFrame as a new column. That's all the pipeline needs to work.

**Step 3 — Build the metric functions in `fairness_metrics.py`**
Write a separate function for each of the four fairness metrics. Each function takes ground truth labels, predicted labels, and a sensitive feature column as inputs, and returns the metric value, its confidence interval, and its status (PASS / WARNING / FLAG). Use fairlearn's `MetricFrame` — it handles the per-group splitting for you.

**Step 4 — Add confidence intervals in `fairness_utils.py`**
Wrap each metric function in a bootstrap loop that runs 1,000 times. On each iteration, resample the test set with replacement and recompute the metric. Take the 2.5th and 97.5th percentiles of the results as your lower and upper bounds, and return the half-width alongside the metric value. This is what lets you say "0.14 ± 0.03" instead of just "0.14."

**Step 5 — Build per-group breakdowns in `subgroup_analysis.py`**
Split the test set by the sensitive feature, compute a confusion matrix for each group, and extract accuracy, TPR (True Positive Rate), FPR (False Positive Rate), and FNR (False Negative Rate) per group. These go directly into the JSON report — make sure your key names match the schema exactly.

**Step 6 — Add the interpretation layer in `bias_detection.py`**
Apply the thresholds from `fairness_utils.py` to each metric value and assign a status. Then generate a plain-English sentence for each flagged or warning result that says which group is affected and what kind of disparity was found. A number alone isn't actionable — the interpretation sentence is what makes the output useful.

**Step 7 — Wire everything together in `fairness_pipeline.py`**
Build a class that runs steps 3–6 in sequence and assembles the full JSON report. The report must match the Standard Output Schema in this README exactly. Add a `save_report()` method that writes the file to `reports/fairness/<timestamp>_<dataset>.json`.

**Step 8 — Write the demo notebook**
In `notebooks/fairness_metrics_validation.ipynb`, show the full flow from start to finish: load Adult Income → train the model → generate predictions → run the pipeline for `sex` → run again for `race` → save both reports → print a readable summary. This notebook is your main Week 4 deliverable.

**Step 9 — Check your numbers against fairlearn**
After your pipeline is working, run fairlearn's built-in `equalized_odds_difference` and `demographic_parity_ratio` functions on the same data and compare results. Your numbers should match within ±0.01. If they don't, there's a bug in your metric logic — find it before moving on. Document this check in the validation notebook.

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

A model can have great overall accuracy while quietly performing much worse for certain groups of people. If you only look at the overall number, you'll miss this completely.

Fairness auditing breaks down model performance by demographic group so those gaps become visible. The goal isn't to prove the model is fair — no set of metrics can do that — but to make sure you know where the disparities are before the model gets used.

**Why this matters:**
A model that correctly identifies 79% of qualified male applicants but only 65% of equally qualified female applicants is not a good model. The overall accuracy number will never show you that.

---

## Metrics

This module computes four complementary fairness metrics. No single metric tells the whole story — you need all four together to get a clear picture.

---

### 1. Equal Opportunity Difference (Primary)

**What it measures:**
The gap in True Positive Rate (TPR) between groups. TPR is the share of people who actually qualify for a positive outcome who the model correctly identifies. A large gap means the model is much better at spotting qualified people in one group than another.

```
Equal Opportunity Difference = TPR(group A) − TPR(group B)
```

**Why it is primary:**
For tasks like income prediction and credit risk, the core fairness question is: are equally qualified people being treated equally? EOD answers that directly — it measures whether qualified individuals in each group are being correctly identified at the same rate.

**Threshold:**

```python
EQUAL_OPPORTUNITY_THRESHOLD = 0.10   # flag if |EOD| > 0.10
```

**Risk levels:**

| Level | Condition | Meaning |
|---|---|---|
| PASS | \|EOD\| ≤ 0.10 | Groups have similar TPR |
| WARNING | 0.10 < \|EOD\| ≤ 0.15 | Moderate gap — worth investigating |
| FLAG | \|EOD\| > 0.15 | Large gap — needs attention before deployment |

**Example output:**

```
Equal Opportunity Difference : 0.14 ± 0.03   ← WARNING
→ Model favors Male group in correctly identifying positives
→ Female group is under-selected despite qualification
```

**Keep in mind:**
The sign of EOD tells you which group is being disadvantaged, not just the size of the gap. Always report which group is affected, not just the number.

---

### 2. Demographic Parity Ratio (Secondary)

**What it measures:**
The ratio of positive prediction rates between two groups. A ratio of 1.0 means both groups receive positive predictions at the same rate. The lower the ratio, the bigger the gap.

```
Demographic Parity Ratio = P(Ŷ=1 | group A) / P(Ŷ=1 | group B)
```

**Why it is secondary:**
Demographic parity sounds intuitive, but it can be misleading when the two groups genuinely have different base rates (i.e., the proportion of truly positive cases differs between groups, as it does in the Adult Income dataset). In that situation, equal prediction rates don't mean equal treatment. Report it for context, but don't try to optimize for it.

**Threshold:**

```python
DEMOGRAPHIC_PARITY_RATIO_THRESHOLD = 0.80   # flag if ratio < 0.80
```

**Risk levels:**

| Level | Condition | Meaning |
|---|---|---|
| PASS | DPR ≥ 0.80 | Prediction rates are broadly similar |
| WARNING | 0.70 ≤ DPR < 0.80 | Borderline — document and monitor |
| FLAG | DPR < 0.70 | Large prediction rate gap |

**Example output:**

```
Demographic Parity Ratio : 0.73   ← Borderline
→ Female group receives positive predictions at 73% the rate of Male group
```

**Keep in mind:**
Always read DPR alongside the base rates section. A DPR below 1.0 doesn't automatically mean something is wrong — it depends on whether the groups genuinely differ in how many truly positive cases they have.

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

Primary benchmark for subgroup fairness evaluation. Predicts whether an individual earns more than $50,000/year. Attributes include age, education, occupation, marital status, race, and gender.

- **Target:** `income` (binary: ≤50K / >50K)
- **Sensitive features:** `sex` (binary), `race` (multi-class — evaluate pairwise)
- **Known consideration:** The dataset reflects 1994 U.S. census data. The
  `sex` column is binary and does not capture gender identity. `race` categories
  are simplified encodings of complex social categories. Acknowledge these
  limitations in the final report.
- **Split:** 80/20 stratified split, fixed random seed for reproducibility

### German Credit Dataset

Secondary benchmark for fairness evaluation in financial decision systems. Predicts whether an individual represents a good or bad credit risk. Attributes include age, gender, employment status, credit history, and loan amount.

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

## Things to Keep in Mind

**You can't optimize all metrics at once.**
Fairness metrics mathematically conflict with each other. It's actually impossible to simultaneously satisfy equal opportunity, demographic parity, and calibration when groups have different base rates (Chouldechova, 2017). Don't try to make every metric pass — pick the one that fits your task and report the others as context.

**These datasets are old and narrow.**
Adult Income is from 1994 U.S. census data. German Credit is from a specific European financial context. Don't generalize findings from these datasets to other populations or use cases without re-running the audit on the new data.

**A passing score doesn't prove fairness.**
These metrics show you where disparities exist. They can't tell you why the disparity is there, whether discrimination occurred, or what to do about it. That part requires understanding the domain and talking to stakeholders.

**The sensitive attributes are simplified.**
Both datasets encode things like gender and race as simple binary or categorical values. Real identity is more complex than that. Be honest about this limitation whenever you write up results from this module.

---

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities.* MIT Press. https://fairmlbook.org/
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments.* Big Data, 5(2).
- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
- fairlearn Documentation: https://fairlearn.org/
- aif360 Documentation: https://aif360.readthedocs.io/
- UCI ML Repository — Adult Income Dataset: https://archive.ics.uci.edu/dataset/2/adult
- UCI ML Repository — German Credit Dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
