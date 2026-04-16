# ML Reliability & Efficiency Toolkit

A modular toolkit for **Responsible AI auditing and LLM context optimization**.

This project explores methods to improve **machine learning reliability, fairness, and efficiency** by combining:

- Responsible AI auditing
- Data leakage detection
- Model evaluation visualization

The goal is to create **practical tool that students can integrate into ML pipelines**.

1. Is the model treating demographic groups fairly?
2. Did the model learn anything it shouldn't have (data leakage)?

---

# Installation & Setup


---

# Project Motivation

In many machine learning workflows, model development focuses heavily on **accuracy**. However, several important risks are often overlooked.

**Bias and Fairness:** Models can perform differently across demographic groups. Without proper auditing, these disparities may go unnoticed until they cause harm.

**Data Leakage:** Models can achieve artificially high performance by accidentally learning information that would not be available at prediction time, leading to failures in production.

**LLM Context Inefficiency:** *(Riddick Input)*

---

# Project Goals

| Module | Goal |
|---|---|
| Responsible AI Auditing | Evaluate model performance across demographic subgroups and compute fairness metrics |
| Data Leakage Detection | Identify potential leakage risks in datasets and ML pipelines |
| Visualization Dashboard | Provide interactive visual summaries of all module outputs |

---

# Repository Structure

```
ml_reliability_efficiency_toolkit/
│
├── modules/
│   ├── fairness/               ← fairness audit module
│   │   └── README.md           ← metrics, thresholds, usage
│   │
│   └── leakage/                ← leakage detection module
│       └── README.md           ← detection methods, risk levels, usage
│
├── dashboard/
│   ├── app.py                  ← Streamlit entry point
│   └── mock_reports/           ← fake JSON reports for dashboard dev
│
├── data/                       ← shared raw datasets
├── docs/                       ← concept explanations, architecture notes
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md                   ← this file
```

---

# Methodology

Each module follows a consistent evaluation approach:

1. **Define formal metrics**
2. **Apply thresholds for interpretation**
3. **Generate structured outputs**
4. **Integrate results into a unified dashboard**

This ensures that results are comparable across modules, interpretable, and actionable.

---

# Technology Stack

| Layer | Libraries |
|---|---|
| **Base** | pandas, numpy, scikit-learn |
| **Fairness Audit** | fairlearn, aif360(optional)|
| **Leakage Detection** | pandas, scikit-learn, great-expectations(optional), scipy |
| **Dashboard** | streamlit, plotly, matplotlib |

---

# Datasets

## Tabular — Fairness Auditing and Leakage Detection

**Adult Income Dataset** (UCI ML Repository): Predicts whether an individual earns more than $50,000/year. Attributes include age, education, occupation, marital status, race, and gender. Used for fairness auditing and subgroup analysis.

**German Credit Dataset**: Predicts whether an individual represents a good or bad credit risk. Attributes include age, gender, employment status, credit history, and loan amount. Used for fairness evaluation in financial decision systems.

**Synthetic Leakage Dataset**: Generated to simulate controlled leakage scenarios including a feature directly encoding the target, future information in training data, and train/test duplication. Used for testing and validating the leakage detection module.


---

# Modules

## 1. Fairness Audit

Evaluates model behavior across demographic groups.

- Computes fairness metrics (e.g., equal opportunity difference, demographic parity ratio)
- Performs subgroup performance analysis
- Flags disparities using defined thresholds

> For metric definitions, thresholds, and evaluation criteria, see [`fairness_audit/README.md`](fairness_audit/README.md).

---

## 2. Data Leakage Detection

Identifies potential sources of leakage in datasets and ML pipelines.

- Detects features strongly correlated with the target
- Identifies train/test contamination
- Highlights high-risk features

> For detection methods, risk levels, and evaluation criteria, see [`leakage_detection/README.md`](leakage_detection/README.md).

---


---

## 4. Visualization Dashboard

Interactive Streamlit application that displays:

- Fairness metrics and subgroup comparisons
- Leakage risk reports
- LLM experiment results

---

# Contributing

1. Work within your assigned module folder
2. Document experiments in the `notebooks/` folder
3. Add explanations and references in the `docs/` folder
4. Use GitHub issues to track tasks and progress

---

# License

For research and educational use.
