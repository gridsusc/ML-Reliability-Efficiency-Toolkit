from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .fairness_utils import (
    FAIRNESS_THRESHOLDS,
    bootstrap_metric,
    compute_group_rates,
    groupwise_metric_frame,
    metric_status,
)

try:
    from fairlearn.metrics import (
        accuracy_score as fl_accuracy_score,
        demographic_parity_ratio as fl_demographic_parity_ratio,
        equalized_odds_difference as fl_equalized_odds_difference,
        false_negative_rate as fl_false_negative_rate,
        false_positive_rate as fl_false_positive_rate,
    )
except Exception:  # pragma: no cover
    from sklearn.metrics import accuracy_score as fl_accuracy_score
    fl_demographic_parity_ratio = None
    fl_equalized_odds_difference = None
    fl_false_negative_rate = None
    fl_false_positive_rate = None


def _as_report(metric_name: str, value: float, lower: float, upper: float) -> dict:
    status = metric_status(metric_name, value, lower, upper)
    return {
        "value": float(value),
        "ci": float((upper - lower) / 2.0) if np.isfinite(lower) and np.isfinite(upper) else float("nan"),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "status": status,
    }


def _bootstrap(metric_name: str, metric_fn: Callable[[Sequence, Sequence, Sequence], float], y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    value, lower, upper, _ = bootstrap_metric(metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)
    return _as_report(metric_name, value, lower, upper)


def equal_opportunity_difference(y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    """Return the equalized-odds-style fairness gap used in the README validation step.

    Fairlearn's equalized_odds_difference is used when available to match the repo's
    evaluation criterion. The return structure is consistent across metrics.
    """
    if fl_equalized_odds_difference is None:
        def metric_fn(yt, yp, sf):
            rates = compute_group_rates(yt, yp, sf)
            tprs = [v["tpr"] for v in rates.values()]
            fprs = [v["fpr"] for v in rates.values()]
            return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))
    else:
        def metric_fn(yt, yp, sf):
            return float(fl_equalized_odds_difference(yt, yp, sensitive_features=sf))
    return _bootstrap("equal_opportunity_difference", metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)


def demographic_parity_ratio(y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    if fl_demographic_parity_ratio is None:
        def metric_fn(yt, yp, sf):
            rates = compute_group_rates(yt, yp, sf)
            selection_rates = [v["selection_rate"] for v in rates.values()]
            mn, mx = min(selection_rates), max(selection_rates)
            return float(mn / mx) if mx else float("nan")
    else:
        def metric_fn(yt, yp, sf):
            return float(fl_demographic_parity_ratio(yt, yp, sensitive_features=sf))
    return _bootstrap("demographic_parity_ratio", metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)


def accuracy_difference(y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    def metric_fn(yt, yp, sf):
        rates = compute_group_rates(yt, yp, sf)
        values = [v["accuracy"] for v in rates.values()]
        return float(max(values) - min(values)) if values else float("nan")

    return _bootstrap("accuracy_difference", metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)


def fpr_difference(y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    def metric_fn(yt, yp, sf):
        rates = compute_group_rates(yt, yp, sf)
        values = [v["fpr"] for v in rates.values()]
        return float(max(values) - min(values)) if values else float("nan")

    return _bootstrap("fpr_difference", metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)


def fnr_difference(y_true, y_pred, sensitive_features, random_state: int = 42) -> dict:
    def metric_fn(yt, yp, sf):
        rates = compute_group_rates(yt, yp, sf)
        values = [v["fnr"] for v in rates.values()]
        return float(max(values) - min(values)) if values else float("nan")

    return _bootstrap("fnr_difference", metric_fn, y_true, y_pred, sensitive_features, random_state=random_state)


METRIC_FUNCTIONS = {
    "equal_opportunity_difference": equal_opportunity_difference,
    "demographic_parity_ratio": demographic_parity_ratio,
    "accuracy_difference": accuracy_difference,
    "fpr_difference": fpr_difference,
    "fnr_difference": fnr_difference,
}
