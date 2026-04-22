"""Fairness module package."""

from .bias_detection import evaluate_metric_status, generate_metric_interpretation
from .fairness_metrics import (
    accuracy_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    fpr_difference,
    fnr_difference,
)
from .fairness_pipeline import FairnessPipeline
from .subgroup_analysis import compute_subgroup_metrics
from .fairness_utils import FAIRNESS_THRESHOLDS

__all__ = [
    "FAIRNESS_THRESHOLDS",
    "FairnessPipeline",
    "accuracy_difference",
    "demographic_parity_ratio",
    "equal_opportunity_difference",
    "evaluate_metric_status",
    "fpr_difference",
    "fnr_difference",
    "generate_metric_interpretation",
    "compute_subgroup_metrics",
]
