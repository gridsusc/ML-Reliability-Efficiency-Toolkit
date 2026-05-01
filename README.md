# ML Reliability & Efficiency Toolkit

A modular toolkit for **Responsible AI auditing and LLM context optimization**.

This project explores methods to improve **machine learning reliability, fairness, and efficiency** by combining:

- Responsible AI auditing
- Data leakage detection
- LLM context efficiency auditing
- Model evaluation visualization

The goal is to create a **practical tool that students can integrate into ML pipelines**.

1. Is the model treating demographic groups fairly?
2. Did the model learn anything it shouldn't have (data leakage)?
3. Is the LLM using its context window efficiently?

---

# Installation & Setup


---

# Project Motivation

In many machine learning workflows, model development focuses heavily on **accuracy**. However, several important risks are often overlooked.

**Bias and Fairness:** Models can perform differently across demographic groups. Without proper auditing, these disparities may go unnoticed until they cause harm.

**Data Leakage:** Models can achieve artificially high performance by accidentally learning information that would not be available at prediction time, leading to failures in production.

**LLM Context Inefficiency:** LLM prompts often waste their context budget on bloated system instructions, redundant retrieved passages, or chunks that are weakly related to the query. This inflates cost and latency, and can degrade response quality — especially in long-context settings, where models exhibit known position-related weaknesses (Liu et al., 2024).

---

# Project Goals

| Module | Goal |
|---|---|
| Responsible AI Auditing | Evaluate model performance across demographic subgroups and compute fairness metrics |
| Data Leakage Detection | Identify potential leakage risks in datasets and ML pipelines |
| Efficiency Auditing | Measure prompt efficiency: token distribution, redundancy, and query–context relevance |
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
│   ├── leakage/                ← leakage detection module
│   │   └── README.md           ← detection methods, risk levels, usage
│   │
│   └── efficiency/             ← LLM context efficiency module
│       └── README.md           ← checks, risk levels, usage
│
├── schemas/                    ← shared Pydantic report models
│   ├── __init__.py
│   ├── base.py                 ← BaseReport (module_name, overall_risk, timestamp, details)
│   ├── fairness.py             ← FairnessReport
│   ├── leakage.py              ← LeakageReport
│   └── efficiency.py           ← EfficiencyReport
│
├── reports/                    ← serialized JSON reports
│   ├── fairness/
│   ├── leakage/
│   └── efficiency/
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

# Architecture

The toolkit follows a three-layer design:

1. **Module layer** — each module (`fairness`, `leakage`, `efficiency`) exposes a `run_audit()` function that takes its inputs as arguments and returns a Pydantic report object. Each module is independently testable and importable.
2. **Schema layer** — a shared `schemas/` package defines a `BaseReport` model and per-module subclasses (`FairnessReport`, `LeakageReport`, `EfficiencyReport`). This is the contract that ties the project together: every module's output conforms to a predictable shape, which is what makes the dashboard possible.
3. **Dashboard layer** — a Streamlit application imports the module functions directly, invokes them, and renders the returned reports. Heavy resources (the sentence-transformer model and tokenizer used by the efficiency module) are loaded once and cached via Streamlit's `@st.cache_resource` decorator.

Reports can also be serialized to JSON files under `reports/<module>/` for archival, debugging, or for testing the dashboard against fixed mock data.

---

# Methodology

Each module follows a consistent evaluation approach:

1. **Define formal metrics**
2. **Apply thresholds for interpretation**
3. **Return a structured Pydantic report (the shared output contract)**
4. **Render results in the unified dashboard**

This ensures that results are comparable across modules, interpretable, and actionable.

---

# Technology Stack

| Layer | Libraries |
|---|---|
| **Base** | pandas, numpy, scikit-learn |
| **Shared Schema** | pydantic |
| **Fairness Audit** | fairlearn |
| **Leakage Detection** | pandas, scikit-learn, scipy, great-expectations (optional) |
| **Efficiency Audit** | tiktoken, sentence-transformers, rank_bm25 |
| **Dashboard** | streamlit, plotly, matplotlib |

---

# Datasets

## Tabular — Fairness Auditing and Leakage Detection

**Adult Income Dataset** (UCI ML Repository): Predicts whether an individual earns more than $50,000/year. Attributes include age, education, occupation, marital status, race, and gender. Used for fairness auditing and subgroup analysis.

**German Credit Dataset**: Predicts whether an individual represents a good or bad credit risk. Attributes include age, gender, employment status, credit history, and loan amount. Used for fairness evaluation in financial decision systems.

**Synthetic Leakage Dataset**: Generated to simulate controlled leakage scenarios including a feature directly encoding the target, future information in training data, and train/test duplication. Used for testing and validating the leakage detection module.

## Text — LLM Context Efficiency

**Natural Questions subset** (Kwiatkowski et al., 2019): A small slice of real Google queries paired with Wikipedia passages. Used for fast validation of the efficiency checks on RAG-style prompts.

**LongBench subset**: A public long-context benchmark, converted into RAG-style prompts via fixed-size chunking and simulated retrieval. Used for robustness testing at realistic context length.

**Synthetic Efficiency Scenarios**: Three hand-built prompts with injected inefficiencies (bloated system prompt, duplicated chunks, irrelevant context). Used as unit tests for the efficiency checks.

---

# Modules

## 1. Fairness Audit

Evaluates model behavior across demographic groups.

- Computes fairness metrics (e.g., equal opportunity difference, demographic parity ratio)
- Performs subgroup performance analysis
- Flags disparities using defined thresholds

> For metric definitions, thresholds, and evaluation criteria, see [`modules/fairness/README.md`](modules/fairness/README.md).

---

## 2. Data Leakage Detection

Identifies potential sources of leakage in datasets and ML pipelines.

- Detects features strongly correlated with the target
- Identifies train/test contamination
- Highlights high-risk features

> For detection methods, risk levels, and evaluation criteria, see [`modules/leakage/README.md`](modules/leakage/README.md).

---

## 3. Efficiency Audit

Evaluates whether an LLM prompt uses its context window efficiently.

- Counts tokens per section to flag budget imbalance (e.g., instruction bloat, over-retrieval)
- Detects redundant chunks via embedding similarity
- Measures semantic relevance between the query and retrieved context
- Aggregates results into an `overall_risk` rating (LOW / MEDIUM / HIGH)

> For checks, risk levels, and evaluation criteria, see [`modules/efficiency/README.md`](modules/efficiency/README.md).

---

## 4. Visualization Dashboard

Interactive Streamlit application that displays:

- Fairness metrics and subgroup comparisons
- Leakage risk reports
- Efficiency audit results

The dashboard imports each module's `run_audit()` function directly and renders the returned Pydantic reports — see the [Architecture](#architecture) section for details.

---

# Contributing

1. Work within your assigned module folder
2. Document experiments in the `notebooks/` folder
3. Add explanations and references in the `docs/` folder
4. Use GitHub issues to track tasks and progress

---

# License

For research and educational use.
