from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .fairness_utils import FAIRNESS_THRESHOLDS

METRIC_PRIORITY = [
    "equal_opportunity_difference",
    "demographic_parity_ratio",
    "fpr_difference",
    "fnr_difference",
    "accuracy_difference",
]


def evaluate_metric_status(metric_name: str, value: float, ci_lower: float, ci_upper: float, threshold: float | None = None) -> str:
    from .fairness_utils import metric_status

    return metric_status(metric_name, value, ci_lower, ci_upper, threshold=threshold)


def _largest_gap_group(metric_name: str, group_values: Dict[str, float]) -> Tuple[str, str, float, float]:
    items = [(g, float(v)) for g, v in group_values.items() if v is not None and not np.isnan(v)]
    if not items:
        return "Unknown", "Unknown", float("nan"), float("nan")

    if metric_name == "demographic_parity_ratio":
        lower_group, lower_value = min(items, key=lambda x: x[1])
        higher_group, higher_value = max(items, key=lambda x: x[1])
        return lower_group, higher_group, lower_value, higher_value

    lower_group, lower_value = min(items, key=lambda x: x[1])
    higher_group, higher_value = max(items, key=lambda x: x[1])
    return lower_group, higher_group, lower_value, higher_value


def generate_metric_interpretation(metric_name: str, metric_result: Dict, subgroup_data: Dict[str, Dict[str, float]]) -> str:
    value = float(metric_result["value"])
    status = metric_result["status"]
    ci_lower = float(metric_result.get("ci_lower", np.nan))
    ci_upper = float(metric_result.get("ci_upper", np.nan))

    if metric_name == "equal_opportunity_difference":
        tpr = subgroup_data.get("subgroup_tpr", {})
        if tpr:
            low_group, high_group, low_val, high_val = _largest_gap_group(metric_name, tpr)
            return (
                f"Model favors {high_group} group in correctly identifying positives; "
                f"{low_group} group is under-selected despite qualification."
            )
        return "Equal opportunity gap detected across groups."

    if metric_name == "demographic_parity_ratio":
        selection = subgroup_data.get("subgroup_selection_rate", {})
        if selection:
            low_group, high_group, low_val, high_val = _largest_gap_group(metric_name, selection)
            ratio = 100.0 * (low_val / high_val) if high_val else float("nan")
            return (
                f"{low_group} group receives positive predictions at {ratio:.1f}% the rate of {high_group} group."
            )
        return "Demographic parity ratio is below the preferred threshold."

    if metric_name == "accuracy_difference":
        accuracy = subgroup_data.get("subgroup_accuracy", {})
        if accuracy:
            low_group, high_group, low_val, high_val = _largest_gap_group(metric_name, accuracy)
            gap = 100.0 * (high_val - low_val)
            return f"{low_group} group has lower accuracy than {high_group} group by {gap:.1f} percentage points."
        return "Accuracy differs across groups."

    if metric_name == "fpr_difference":
        fpr = subgroup_data.get("subgroup_fpr", {})
        if fpr:
            low_group, high_group, low_val, high_val = _largest_gap_group(metric_name, fpr)
            gap = 100.0 * (high_val - low_val)
            return f"{high_group} group experiences elevated false positive rate by {gap:.1f} percentage points relative to {low_group} group."
        return "False positive rate disparity detected."

    if metric_name == "fnr_difference":
        fnr = subgroup_data.get("subgroup_fnr", {})
        if fnr:
            low_group, high_group, low_val, high_val = _largest_gap_group(metric_name, fnr)
            gap = 100.0 * (high_val - low_val)
            return f"{high_group} group experiences elevated false negative rate by {gap:.1f} percentage points relative to {low_group} group."
        return "False negative rate disparity detected."

    return f"{metric_name} = {value:.4f} ({status})."


def summarize_report_status(metrics: Dict[str, Dict[str, float]]) -> str:
    statuses = [m.get("status", "PASS") for m in metrics.values()]
    if any(s == "FLAG" for s in statuses):
        return "FAIL"
    if any(s == "WARNING" for s in statuses):
        return "WARNING"
    return "PASS"


def collect_flags(metrics: Dict[str, Dict[str, float]]) -> List[str]:
    return [name for name, info in metrics.items() if info.get("status") in {"WARNING", "FLAG"}]

