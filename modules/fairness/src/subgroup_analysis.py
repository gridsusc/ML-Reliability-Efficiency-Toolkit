from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .fairness_utils import compute_group_rates, validate_inputs


def compute_subgroup_metrics(y_true: Sequence, y_pred: Sequence, sensitive_features: Sequence) -> Dict[str, Dict[str, float]]:
    """Compute per-group accuracy, TPR, FPR, and FNR for the report schema."""
    validate_inputs(y_true, y_pred, sensitive_features)
    group_rates = compute_group_rates(y_true, y_pred, sensitive_features)

    subgroup_accuracy: Dict[str, float] = {}
    subgroup_tpr: Dict[str, float] = {}
    subgroup_fpr: Dict[str, float] = {}
    subgroup_fnr: Dict[str, float] = {}
    support: Dict[str, int] = {}

    for group, values in group_rates.items():
        subgroup_accuracy[group] = round(float(values["accuracy"]), 6)
        subgroup_tpr[group] = round(float(values["tpr"]), 6)
        subgroup_fpr[group] = round(float(values["fpr"]), 6)
        subgroup_fnr[group] = round(float(values["fnr"]), 6)
        support[group] = int(values["support"])

    return {
        "subgroup_accuracy": subgroup_accuracy,
        "subgroup_tpr": subgroup_tpr,
        "subgroup_fpr": subgroup_fpr,
        "subgroup_fnr": subgroup_fnr,
        "support": support,
    }
