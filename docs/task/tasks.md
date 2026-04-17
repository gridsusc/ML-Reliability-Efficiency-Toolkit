# Student Task List — ML Reliability & Efficiency Toolkit

One to-do list per student. Tasks are ordered so each builds on the previous one — don't skip ahead. Check off items as you complete them.

---

## Student A — Fairness Module Owner

### Setup
- [ ] Clone the repo and create a `student-a/fairness` branch
- [ ] Create a virtual environment and run `pip install -r requirements.txt`
- [ ] Verify `import pandas, sklearn, fairlearn` works without errors
- [ ] Download the Adult Income dataset from UCI into `data/`
- [ ] Confirm you can load it: a 3-line script that prints `df.shape` and `df['income'].value_counts()`

### Data preparation
- [ ] Handle missing values in Adult Income
- [ ] Encode categorical columns (keep `sex` and `race` clean and consistently labeled)
- [ ] Separate the target (`income`) from the features
- [ ] Create an 80/20 stratified train/test split with a fixed random seed
- [ ] Save the cleaned dataset to `modules/fairness/data/processed/`

### Baseline model
- [ ] Train a logistic regression on the training split using non-sensitive features only (no `sex`, no `race` as inputs)
- [ ] Generate predictions on the test split
- [ ] Attach predictions to the test DataFrame as a new column

### Metric functions (`src/fairness_metrics.py`)
- [ ] Write the Equal Opportunity Difference function using fairlearn's `MetricFrame`
- [ ] Write the Accuracy Difference function
- [ ] Each function returns `{"value": float, "status": "PASS" | "WARNING" | "FLAG"}`
- [ ] Thresholds come from `src/fairness_utils.py`, not hardcoded

### Per-group breakdowns (`src/subgroup_analysis.py`)
- [ ] Split the test set by sensitive feature
- [ ] Compute confusion matrix per group
- [ ] Extract accuracy, TPR, FPR, FNR per group
- [ ] Return as a dict with key names matching the JSON schema exactly

### Pipeline (`src/fairness_pipeline.py`)
- [ ] Create `FairnessPipeline` class with `run_all_metrics()` method
- [ ] Assemble the full JSON report matching the Standard Output Schema in the README
- [ ] Add `save_report()` method that writes to `outputs/reports/<timestamp>_<dataset>.json`
- [ ] Status field correctly set: any FLAG → FAIL, any WARNING → WARNING, else PASS

### Demo notebook (`notebooks/fairness_demo.ipynb`)
- [ ] Load Adult Income
- [ ] Train model, generate predictions
- [ ] Run `FairnessPipeline` for `sex`
- [ ] Save the JSON report
- [ ] Print a readable summary of the results
- [ ] Notebook runs top-to-bottom with no errors

### Sanity check
- [ ] Call fairlearn's `equalized_odds_difference` on the same data
- [ ] Confirm your EOD matches within ±0.01
- [ ] If it doesn't, fix the bug before moving on
- [ ] Document the check in the notebook

### Dashboard integration
- [ ] Generate a real fairness JSON report
- [ ] Coordinate with Student C to make sure the dashboard loads it
- [ ] Take a screenshot of the dashboard showing real fairness data
- [ ] Attach the screenshot to a PR

### Final polish
- [ ] Fill in any gaps in the demo notebook
- [ ] Confirm the fairness README reflects what you actually built
- [ ] Add a real example output block in the README (real JSON + dashboard screenshot)

---

## Student B — Leakage Module Owner

### Setup
- [ ] Clone the repo and create a `student-b/leakage` branch
- [ ] Create a virtual environment and run `pip install -r requirements.txt`
- [ ] Verify `import pandas, sklearn, scipy` works without errors
- [ ] Download the Adult Income dataset into `data/`
- [ ] Confirm you can load it and print `df.shape`

### Data preparation
- [ ] Handle missing values in Adult Income
- [ ] Encode categorical columns
- [ ] Separate target from features
- [ ] Create an 80/20 stratified train/test split with a fixed random seed
- [ ] Keep train and test as separate DataFrames — do not merge them again
- [ ] Save the cleaned dataset to `modules/leakage/data/processed/`

### Synthetic validation dataset (build this FIRST)
- [ ] In `data/synthetic/`, build Scenario 1: direct encoding (feature is a noisy copy of the target)
- [ ] In `data/synthetic/`, build Scenario 2: contamination (5% of test rows duplicated from train)
- [ ] Each scenario is a function that returns `(train_df, test_df, target_col)`
- [ ] Each scenario prints its shape and first 5 rows as a sanity check

### Shared helpers (`src/leakage_utils.py`)
- [ ] Define `LEAKAGE_THRESHOLDS` dict (copy from the README)
- [ ] Add row-hashing helper function
- [ ] Any other helpers you reuse across checks go here
- [ ] Each check imports thresholds from this file — do not hardcode values inside checks

### Check 1 — Target Correlation (`src/target_correlation.py`)
- [ ] For each feature, compute Pearson correlation (numeric) or Spearman (encoded categorical) with the target
- [ ] Assign severity (LOW / MEDIUM / HIGH) based on thresholds
- [ ] Return correlation values + severity per feature
- [ ] Test on the direct-encoding synthetic scenario → must return HIGH for the encoded column
- [ ] Test on clean Adult Income → no HIGH flags (false-positive discipline)

### Check 2 — Train/Test Contamination (`src/train_test_contamination.py`)
- [ ] Hash every row in train and test (all feature columns, not the target)
- [ ] Count overlapping hashes
- [ ] Compute contamination ratio = `duplicate_rows / len(df_test)`
- [ ] Assign severity based on ratio
- [ ] Test on the contamination synthetic scenario → must return HIGH
- [ ] Test on clean Adult Income → LOW

### Check 3 — Future Information (`src/future_information.py`)
- [ ] Accept optional `datetime_cols` list and `reference_date` parameters
- [ ] If either is missing, return SKIPPED (not LOW or PASS)
- [ ] For each datetime column, flag values that postdate `reference_date`
- [ ] Assign severity based on the fraction and pattern of future-dated values

### Pipeline (`src/leakage_pipeline.py`)
- [ ] Create `LeakagePipeline` class that takes train DataFrame, test DataFrame, target column, optional temporal params
- [ ] Run all three checks in sequence
- [ ] Compute `overall_risk` using the aggregation rule (any HIGH → HIGH; any MEDIUM → MEDIUM; else LOW)
- [ ] Assemble the full JSON report matching the Standard Output Schema in the README
- [ ] Add `save_report()` method that writes to `outputs/reports/<timestamp>_<dataset>.json`

### Demo notebook (`notebooks/leakage_demo.ipynb`)
- [ ] Section 1: Run on clean Adult Income → verify `overall_risk = LOW`
- [ ] Section 2: Run on direct-encoding scenario → verify Check 1 returns HIGH
- [ ] Section 3: Run on contamination scenario → verify Check 2 returns HIGH
- [ ] Print a readable summary in each section
- [ ] Notebook runs top-to-bottom with no errors

### Dashboard integration
- [ ] Generate a real leakage JSON report from clean Adult Income
- [ ] Coordinate with Student C to make sure the dashboard loads it
- [ ] Take a screenshot of the dashboard showing real leakage data
- [ ] Attach the screenshot to a PR

### Final polish
- [ ] Fill in any gaps in the demo notebook
- [ ] Confirm the leakage README reflects what you actually built
- [ ] Add a real example output block in the README

---

## Student C — Dashboard and Integration Lead

### Setup
- [ ] Clone the repo and create a `student-c/dashboard` branch
- [ ] Create a virtual environment and run `pip install -r requirements.txt`
- [ ] Verify `import streamlit, plotly, pandas` works without errors
- [ ] Confirm `streamlit --version` runs

### JSON output contract (do this FIRST, before anyone else starts)
- [ ] Read both module READMEs and extract the exact JSON schema for each
- [ ] Write a mock fairness JSON report → save to `dashboard/mock_reports/fairness_mock.json`
- [ ] Write a mock leakage JSON report → save to `dashboard/mock_reports/leakage_mock.json`
- [ ] Both mocks match the schemas in the module READMEs exactly
- [ ] Open a PR with the mocks; get Student A and Student B to approve them

### Dashboard skeleton (`dashboard/app.py`)
- [ ] Create a minimal Streamlit app that launches without errors
- [ ] Add two tabs using `st.tabs(["Fairness", "Leakage"])`
- [ ] Each tab loads its mock JSON file and displays the `status` + `summary` fields
- [ ] Color-code status: green PASS, yellow WARNING, red FAIL

### Fairness tab (rendering real data)
- [ ] Display EOD value with its status badge
- [ ] Display Accuracy Difference value with its status badge
- [ ] Build a Plotly grouped bar chart: one bar per demographic group, one color per metric (accuracy, TPR, FPR, FNR)
- [ ] Display the base rates per group
- [ ] All rendering reads from the JSON file — no hardcoded values

### Leakage tab (rendering real data)
- [ ] Display `overall_risk` with a big color-coded badge
- [ ] Display the summary sentence
- [ ] Build a table with one row per check showing risk level (color-coded: green LOW, yellow MEDIUM, red HIGH)
- [ ] For each flagged check, show the list of flagged features or columns
- [ ] All rendering reads from the JSON file — no hardcoded values

### Report loading utility
- [ ] Write a `load_latest_report(module_name)` function that finds the most recent JSON report per module (by filename timestamp)
- [ ] Function looks in `modules/<module>/outputs/reports/` for real reports
- [ ] Falls back to `dashboard/mock_reports/` if no real report exists
- [ ] Dashboard uses this utility — not hardcoded paths
- [ ] Show "Last updated: YYYY-MM-DD HH:MM" per module at the top of each tab

### Integration with real modules
- [ ] After Student A generates a real fairness report, confirm your tab renders it correctly
- [ ] After Student B generates a real leakage report, confirm your tab renders it correctly
- [ ] Fix any schema mismatches immediately — coordinate with A and B to resolve them

### Presentation deck
- [ ] One slide per module (chart + one-sentence finding)
- [ ] One architecture slide (repo structure + JSON contract)
- [ ] One live demo slide (walkthrough of the dashboard)
- [ ] One takeaways slide
- [ ] Rehearse the live demo once with the full team
- [ ] Save slides to `docs/presentation/`

---

## Student D — Testing, Documentation, and Evaluation Lead

### Setup
- [ ] Clone the repo and create a `student-d/infra` branch
- [ ] Create a virtual environment and run `pip install -r requirements.txt`
- [ ] Verify `pytest --version` runs
- [ ] Verify `jupyter notebook` launches

### Repo hygiene (do this FIRST)
- [ ] Confirm the current repo structure matches the target in the main README
- [ ] Delete any legacy folders still hanging around (`evaluation/`, root `notebooks/`, root `reports/`)
- [ ] Confirm `requirements.txt` has real pinned dependencies (not the word "Requirements")
- [ ] Confirm `pytest.ini` exists at repo root
- [ ] Run `pytest` — should execute even if all tests are skipped

### Kickoff documentation
- [ ] Write kickoff meeting notes in `docs/kickoff.md`
- [ ] Document who owns each module
- [ ] Document the JSON output contract agreement
- [ ] Document any decisions made (e.g., dropped LLM module, scoped-down metrics)

### Notebook scaffolds (for A and B to fill in)
- [ ] Create `modules/fairness/notebooks/fairness_demo.ipynb` with 5 labeled sections: Load Data, Train Model, Compute Metrics, Save Report, Summary
- [ ] Each section has a placeholder markdown cell and an empty code cell
- [ ] Create `modules/leakage/notebooks/leakage_demo.ipynb` with 3 labeled sections: Clean Adult Income, Direct Encoding Validation, Contamination Validation
- [ ] Both notebooks open in Jupyter without errors

### Fairness tests (`modules/fairness/tests/test_fairness_metrics.py`)
- [ ] Implement `test_eod_matches_fairlearn`: assert our EOD matches fairlearn's `equalized_odds_difference` within ±0.01
- [ ] Implement `test_subgroup_metrics_are_internally_consistent`: assert TPR + FNR = 1.0 per group, FPR + TNR = 1.0 per group
- [ ] Run `pytest modules/fairness/tests/` — both tests pass

### Leakage tests (`modules/leakage/tests/test_leakage_checks.py`)
- [ ] Implement `test_target_correlation_fires_on_direct_encoding`: load synthetic scenario 1, assert Check 1 returns HIGH
- [ ] Implement `test_contamination_fires_on_duplicated_test_rows`: load synthetic scenario 2, assert Check 2 returns HIGH and ratio ≥ 0.05
- [ ] Implement `test_clean_data_returns_low_risk`: run on clean Adult Income, assert `overall_risk == "LOW"`
- [ ] Run `pytest modules/leakage/tests/` — all three tests pass

### End-to-end integration test
- [ ] Clone the repo fresh in a new environment
- [ ] Follow the main README install steps exactly
- [ ] Run both demo notebooks top-to-bottom
- [ ] Launch the dashboard with `streamlit run dashboard/app.py`
- [ ] Verify both tabs render real data
- [ ] Take a screenshot of the working dashboard
- [ ] Save screenshot + notes to `docs/integration_test.md`
- [ ] File any bugs found as GitHub issues

### Final repo audit
- [ ] Check that all READMEs are internally consistent (no contradictions between main README and module READMEs)
- [ ] Check that all internal links work (no broken `modules/fairness/README.md` paths)
- [ ] Confirm the install steps in the main README actually work on a fresh clone
- [ ] Remove any dead code or unused files
- [ ] Document the audit results in `docs/final_audit.md`

---

## Cross-Team Shared Tasks

These are owned by no single student but happen with the full team.

- [ ] Kickoff meeting: lock module ownership, confirm JSON contract, set up GitHub Project board
- [ ] Mid-project sync: each student shows their current state, blockers surfaced, schema mismatches resolved
- [ ] Dashboard integration session: all four students in the same call, debug any JSON schema mismatches in real time
- [ ] Code freeze: no new features after this point, only bug fixes
- [ ] Presentation rehearsal: full run-through of the deck and live demo with the whole team
