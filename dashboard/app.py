"""
ML Reliability & Efficiency Toolkit — Dashboard

Streamlit entry point that visualizes fairness audit and leakage detection
reports produced by the toolkit's modules.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import glob

import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "mock_reports")


def load_reports(directory: str) -> dict[str, dict]:
    """Load all JSON report files from *directory* and return them keyed by
    filename (without extension).

    Each JSON file is expected to contain a top-level object.  Fairness
    reports should include ``metrics``, ``subgroups``, and ``flags`` keys.
    Leakage reports should include ``summary``, ``flagged_features``, and
    ``notes`` keys.  See ``mock_reports/`` for examples.
    """
    reports: dict[str, dict] = {}
    try:
        paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    except OSError:
        return reports
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                reports[name] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
    return reports


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Reliability & Efficiency Toolkit",
    page_icon="🔍",
    layout="wide",
)

st.title("ML Reliability & Efficiency Toolkit")
st.markdown(
    "Interactive dashboard for **fairness audit** and **leakage detection** reports."
)

# ---------------------------------------------------------------------------
# Sidebar — report source selector
# ---------------------------------------------------------------------------

st.sidebar.header("Report Source")
use_mock = st.sidebar.checkbox("Use mock reports (for development)", value=True)

if use_mock:
    report_dir = MOCK_REPORTS_DIR
else:
    report_dir = st.sidebar.text_input(
        "Path to report directory",
        value="",
        help="Absolute or relative path to a directory containing JSON report files.",
    )

if not report_dir or not os.path.isdir(report_dir):
    st.info("👈 Select a valid report directory to get started.")
    st.stop()

reports = load_reports(report_dir)

if not reports:
    st.warning("No JSON reports found in the selected directory.")
    st.stop()

# ---------------------------------------------------------------------------
# Fairness section
# ---------------------------------------------------------------------------

fairness_reports = {k: v for k, v in reports.items() if "fairness" in k}
leakage_reports = {k: v for k, v in reports.items() if "leakage" in k}

st.header("Fairness Audit")

if fairness_reports:
    for name, report in fairness_reports.items():
        with st.expander(f"📄 {name}", expanded=True):
            # Summary metrics
            if "metrics" in report:
                cols = st.columns(len(report["metrics"]))
                for col, (metric, value) in zip(cols, report["metrics"].items(), strict=True):
                    col.metric(label=metric, value=f"{value:.4f}")

            # Subgroup details
            if "subgroups" in report:
                st.subheader("Subgroup Breakdown")
                st.table(report["subgroups"])

            # Flags
            if "flags" in report:
                st.subheader("Flags")
                for flag in report["flags"]:
                    st.warning(flag)
else:
    st.info("No fairness reports available.")

# ---------------------------------------------------------------------------
# Leakage section
# ---------------------------------------------------------------------------

st.header("Leakage Detection")

if leakage_reports:
    for name, report in leakage_reports.items():
        with st.expander(f"📄 {name}", expanded=True):
            # Summary
            if "summary" in report:
                st.subheader("Summary")
                summary = report["summary"]
                cols = st.columns(3)
                cols[0].metric("Features Checked", summary.get("features_checked", "—"))
                cols[1].metric("Flags Raised", summary.get("flags_raised", "—"))
                cols[2].metric("Risk Level", summary.get("risk_level", "—"))

            # Flagged features
            if "flagged_features" in report:
                st.subheader("Flagged Features")
                st.table(report["flagged_features"])

            # Notes
            if "notes" in report:
                st.subheader("Notes")
                for note in report["notes"]:
                    st.info(note)
else:
    st.info("No leakage reports available.")
