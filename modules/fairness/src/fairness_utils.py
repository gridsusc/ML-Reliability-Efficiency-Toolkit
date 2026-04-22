from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from fairlearn.metrics import (
        MetricFrame,
        accuracy_score as fl_accuracy_score,
        demographic_parity_ratio as fl_demographic_parity_ratio,
        equalized_odds_difference as fl_equalized_odds_difference,
        false_negative_rate as fl_false_negative_rate,
        false_positive_rate as fl_false_positive_rate,
        selection_rate as fl_selection_rate,
        true_positive_rate as fl_true_positive_rate,
    )
except Exception:  # pragma: no cover - fallback for environments without fairlearn
    MetricFrame = None
    fl_accuracy_score = accuracy_score
    fl_demographic_parity_ratio = None
    fl_equalized_odds_difference = None
    fl_false_negative_rate = None
    fl_false_positive_rate = None
    fl_selection_rate = None
    fl_true_positive_rate = None


FAIRNESS_THRESHOLDS: Dict[str, float] = {
    "equal_opportunity_difference": 0.10,
    "demographic_parity_ratio": 0.80,
    "accuracy_difference": 0.05,
    "fpr_difference": 0.10,
    "fnr_difference": 0.10,
}

WARNING_MULTIPLIER = 1.5
BOOTSTRAP_ITERATIONS = 1000
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class MetricResult:
    value: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float
    status: str

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "value": float(self.value),
            "ci": float(self.ci_half_width),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "status": self.status,
        }


def _to_numpy(values: Sequence) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index)):
        return values.to_numpy()
    return np.asarray(values)


def validate_binary_labels(y_true: Sequence, y_pred: Sequence) -> None:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    unique_true = set(pd.Series(yt).dropna().unique().tolist())
    unique_pred = set(pd.Series(yp).dropna().unique().tolist())
    if not unique_true.issubset({0, 1}) or not unique_pred.issubset({0, 1}):
        raise ValueError("y_true and y_pred must be binary labels encoded as 0/1")


def validate_inputs(y_true: Sequence, y_pred: Sequence, sensitive_features: Sequence) -> None:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    sf = _to_numpy(sensitive_features)
    if not (len(yt) == len(yp) == len(sf)):
        raise ValueError("y_true, y_pred, and sensitive_features must have the same length")
    validate_binary_labels(yt, yp)


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def compute_group_rates(
    y_true: Sequence,
    y_pred: Sequence,
    sensitive_features: Sequence,
) -> Dict[str, Dict[str, float]]:
    """Return per-group accuracy, TPR, FPR, FNR, selection rate, and base rate."""
    validate_inputs(y_true, y_pred, sensitive_features)
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)
    sf = _to_numpy(sensitive_features)

    groups = pd.Series(sf).astype("object")
    results: Dict[str, Dict[str, float]] = {}

    for group_name in pd.unique(groups):
        mask = groups == group_name
        ytg = yt[mask]
        ypg = yp[mask]
        tn, fp, fn, tp = confusion_matrix(ytg, ypg, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        accuracy = safe_divide(tp + tn, total)
        tpr = safe_divide(tp, tp + fn)
        fpr = safe_divide(fp, fp + tn)
        fnr = safe_divide(fn, fn + tp)
        selection_rate = safe_divide(np.sum(ypg == 1), total)
        base_rate = safe_divide(np.sum(ytg == 1), total)
        results[str(group_name)] = {
            "accuracy": float(accuracy),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "selection_rate": float(selection_rate),
            "base_rate": float(base_rate),
            "support": int(total),
        }
    return results


def compute_base_rates(y_true: Sequence, sensitive_features: Sequence) -> Dict[str, float]:
    yt = _to_numpy(y_true).astype(int)
    sf = _to_numpy(sensitive_features)
    if len(yt) != len(sf):
        raise ValueError("y_true and sensitive_features must have the same length")
    groups = pd.Series(sf).astype("object")
    base_rates: Dict[str, float] = {}
    for group_name in pd.unique(groups):
        mask = groups == group_name
        base_rates[str(group_name)] = float(np.mean(yt[mask])) if mask.any() else float("nan")
    return base_rates


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def bootstrap_metric(
    metric_fn: Callable[[Sequence, Sequence, Sequence], float],
    y_true: Sequence,
    y_pred: Sequence,
    sensitive_features: Sequence,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_attempts: int | None = None,
) -> Tuple[float, float, float, float]:
    """Return metric value and a percentile bootstrap CI half-width.

    The metric is recomputed on bootstrap-resampled test sets.
    """
    validate_inputs(y_true, y_pred, sensitive_features)
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    sf = _to_numpy(sensitive_features)
    n = len(yt)
    rng = np.random.default_rng(random_state)

    value = float(metric_fn(yt, yp, sf))
    samples: list[float] = []
    attempts = 0
    limit = max_attempts if max_attempts is not None else n_bootstrap * 5

    while len(samples) < n_bootstrap and attempts < limit:
        attempts += 1
        idx = _bootstrap_indices(n, rng)
        try:
            boot_value = float(metric_fn(yt[idx], yp[idx], sf[idx]))
        except Exception:
            continue
        if np.isfinite(boot_value):
            samples.append(boot_value)

    if not samples:
        return value, float("nan"), float("nan"), float("nan")

    arr = np.asarray(samples, dtype=float)
    lower = float(np.percentile(arr, 2.5))
    upper = float(np.percentile(arr, 97.5))
    half_width = float((upper - lower) / 2.0)
    return value, lower, upper, half_width


def _status_from_bounds(
    value: float,
    lower: float,
    upper: float,
    threshold: float,
    metric_type: str,
) -> str:
    if np.isnan(value):
        return "FLAG"

    if metric_type == "ratio":
        if value >= threshold:
            return "PASS"
        warning_cutoff = threshold * (1.0 - (1.0 - 0.875))  # 0.875 * threshold
        if value >= warning_cutoff:
            return "WARNING"
        return "FLAG"

    abs_value = abs(value)
    if abs_value <= threshold:
        return "PASS"
    if abs_value <= threshold * WARNING_MULTIPLIER:
        return "WARNING"
    return "FLAG"


def metric_status(
    metric_name: str,
    value: float,
    ci_lower: float,
    ci_upper: float,
    threshold: float | None = None,
) -> str:
    """Assign PASS/WARNING/FLAG using README thresholds and CI overlap."""
    if threshold is None:
        threshold = FAIRNESS_THRESHOLDS[metric_name]

    metric_type = "ratio" if metric_name == "demographic_parity_ratio" else "difference"
    status = _status_from_bounds(value, ci_lower, ci_upper, threshold, metric_type)

    # If the point estimate is okay but the confidence interval crosses the threshold,
    # escalate to WARNING so the report highlights uncertainty.
    if metric_type == "difference":
        if abs(value) <= threshold and (abs(ci_lower) <= threshold <= abs(ci_upper) or abs(ci_upper) <= threshold <= abs(ci_lower)):
            return "WARNING"
        if status == "FLAG" and min(abs(ci_lower), abs(ci_upper)) <= threshold:
            return "WARNING"
    else:
        if value >= threshold and ci_lower < threshold < ci_upper:
            return "WARNING"
        if status == "FLAG" and ci_upper >= threshold:
            return "WARNING"
    return status


def groupwise_metric_frame(
    y_true: Sequence,
    y_pred: Sequence,
    sensitive_features: Sequence,
) -> "MetricFrame | Dict[str, Dict[str, float]]":
    """Return a fairlearn MetricFrame when available, otherwise a lightweight fallback."""
    validate_inputs(y_true, y_pred, sensitive_features)
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    sf = _to_numpy(sensitive_features)

    if MetricFrame is None:
        return compute_group_rates(yt, yp, sf)

    metrics = {
        "accuracy": fl_accuracy_score,
        "tpr": fl_true_positive_rate,
        "fpr": fl_false_positive_rate,
        "fnr": fl_false_negative_rate,
        "selection_rate": fl_selection_rate,
    }
    return MetricFrame(metrics=metrics, y_true=yt, y_pred=yp, sensitive_features=sf)

