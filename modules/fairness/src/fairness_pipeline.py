from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)


@dataclass
class FairnessThresholds:
    pass_eod: float = 0.10
    warn_eod: float = 0.15
    pass_di: float = 0.80


class FairnessPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        prediction_col: str,
        sensitive_feature: str,
        dataset: str = "unknown",
        positive_label: Any = 1,
        thresholds: FairnessThresholds | None = None,
        random_state: int = 42,
        model: Any | None = None,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.sensitive_feature = sensitive_feature
        self.dataset = dataset
        self.positive_label = positive_label
        self.thresholds = thresholds or FairnessThresholds()
        self.random_state = random_state
        self.model = model

        self._validate_inputs()
        self._normalize_binary_columns()

    def _validate_inputs(self) -> None:
        if self.df is None or self.df.empty:
            raise ValueError("Input dataframe is empty.")

        required = [self.target_col, self.prediction_col, self.sensitive_feature]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _to_binary(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = set(pd.Series(series.dropna().unique()).tolist())
            if unique_vals.issubset({0, 1}):
                return series.fillna(0).astype(int)
            return (series.fillna(0) >= 0.5).astype(int)

        truthy = {
            str(self.positive_label).strip().lower(),
            "1",
            "true",
            "yes",
            "y",
            "positive",
            "pos",
        }
        return series.astype(str).str.strip().str.lower().isin(truthy).astype(int)

    def _normalize_binary_columns(self) -> None:
        self.df[self.target_col] = self._to_binary(self.df[self.target_col])
        self.df[self.prediction_col] = self._to_binary(self.df[self.prediction_col])

    def _bootstrap_ci(
        self,
        metric_fn: Callable[[pd.DataFrame], float],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        values: List[float] = []
        n = len(self.df)

        if n == 0:
            return 0.0, 0.0

        rng = np.random.default_rng(self.random_state)

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            sample = self.df.iloc[sample_idx].reset_index(drop=True)
            values.append(float(metric_fn(sample)))

        lower = float(np.percentile(values, 100 * (alpha / 2)))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        return lower, upper

    def _binary_group_metrics(self, group_df: pd.DataFrame) -> Dict[str, float]:
        y_true = group_df[self.target_col].to_numpy()
        y_pred = group_df[self.prediction_col].to_numpy()

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        total = max(len(group_df), 1)
        pos = max(tp + fn, 1)
        neg = max(tn + fp, 1)

        return {
            "count": int(len(group_df)),
            "accuracy": float((tp + tn) / total),
            "tpr": float(tp / pos),
            "fpr": float(fp / neg),
            "fnr": float(fn / pos),
            "selection_rate": float((y_pred == 1).mean()) if len(group_df) else 0.0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def _group_breakdown(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for group_value in data[self.sensitive_feature].dropna().unique():
            group_df = data[data[self.sensitive_feature] == group_value]
            results[str(group_value)] = self._binary_group_metrics(group_df)
        return results

    def _fairness_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        y_true = data[self.target_col]
        y_pred = data[self.prediction_col]
        sensitive = data[self.sensitive_feature]

        eod = float(
            equal_opportunity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        )
        dpd = float(
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        )
        eodds = float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        )
        dpr = float(
            demographic_parity_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        )

        rates = []
        for g in pd.Series(sensitive).dropna().unique():
            mask = sensitive == g
            rate = float((y_pred[mask] == 1).mean()) if mask.any() else 0.0
            rates.append(rate)

        disparate_impact = float(min(rates) / max(max(rates), 1e-12)) if rates else 0.0

        return {
            "equal_opportunity_difference": eod,
            "demographic_parity_difference": dpd,
            "equalized_odds_difference": eodds,
            "demographic_parity_ratio": dpr,
            "disparate_impact": disparate_impact,
        }

    def _status_from_metrics(self, metrics: Dict[str, float]) -> str:
        eod = metrics["equal_opportunity_difference"]
        di = metrics["disparate_impact"]

        if eod > self.thresholds.warn_eod or di < self.thresholds.pass_di:
            return "FLAG"
        if eod > self.thresholds.pass_eod:
            return "WARNING"
        return "PASS"

    def _collect_flags(self, metrics: Dict[str, float]) -> List[str]:
        flags: List[str] = []

        if metrics["equal_opportunity_difference"] > self.thresholds.pass_eod:
            flags.append("Equal opportunity difference exceeds pass threshold.")
        if metrics["disparate_impact"] < self.thresholds.pass_di:
            flags.append("Disparate impact is below acceptable range.")
        if metrics["equalized_odds_difference"] > self.thresholds.pass_eod:
            flags.append("Equalized odds difference is elevated.")

        return flags

    def _build_summary(
        self,
        metrics: Dict[str, float],
        subgroup_data: Dict[str, Dict[str, float]],
    ) -> str:
        eod = metrics["equal_opportunity_difference"]
        di = metrics["disparate_impact"]

        best_group = None
        worst_group = None
        if subgroup_data:
            by_tpr = sorted(subgroup_data.items(), key=lambda item: item[1]["tpr"])
            worst_group = by_tpr[0][0]
            best_group = by_tpr[-1][0]

        if eod <= self.thresholds.pass_eod:
            base = f"Fairness looks acceptable. Equal opportunity difference is {eod:.3f}."
        elif eod <= self.thresholds.warn_eod:
            base = f"Fairness needs review. Equal opportunity difference is {eod:.3f}."
        else:
            base = f"Fairness risk is high. Equal opportunity difference is {eod:.3f}."

        if best_group is not None and worst_group is not None:
            base += f" Best TPR group: {best_group}. Worst TPR group: {worst_group}."

        base += f" Disparate impact is {di:.3f}."
        return base

    def _build_recommendations(
        self,
        metrics: Dict[str, float],
        subgroup_data: Dict[str, Dict[str, float]],
    ) -> List[str]:
        recs: List[str] = []

        if metrics["equal_opportunity_difference"] > self.thresholds.pass_eod:
            recs.append("Review thresholding or apply post-processing to reduce TPR gaps.")
        if metrics["disparate_impact"] < self.thresholds.pass_di:
            recs.append("Check whether one group is receiving systematically fewer positive predictions.")
        if metrics["equalized_odds_difference"] > self.thresholds.pass_eod:
            recs.append("Investigate both TPR and FPR differences across groups.")
        if not recs:
            recs.append("No immediate fairness mitigation is required.")

        if subgroup_data:
            worst = min(subgroup_data.items(), key=lambda item: item[1]["tpr"])[0]
            best = max(subgroup_data.items(), key=lambda item: item[1]["tpr"])[0]
            if worst != best:
                recs.append(f"Compare decision patterns for group {worst} against {best}.")

        return recs

    def run_all_metrics(self) -> Dict[str, Any]:
        metrics = self._fairness_metrics(self.df)
        subgroup_data = self._group_breakdown(self.df)

        eod_ci = self._bootstrap_ci(
            lambda d: float(
                equal_opportunity_difference(
                    y_true=d[self.target_col],
                    y_pred=d[self.prediction_col],
                    sensitive_features=d[self.sensitive_feature],
                )
            )
        )
        dpd_ci = self._bootstrap_ci(
            lambda d: float(
                demographic_parity_difference(
                    y_true=d[self.target_col],
                    y_pred=d[self.prediction_col],
                    sensitive_features=d[self.sensitive_feature],
                )
            )
        )
        eodds_ci = self._bootstrap_ci(
            lambda d: float(
                equalized_odds_difference(
                    y_true=d[self.target_col],
                    y_pred=d[self.prediction_col],
                    sensitive_features=d[self.sensitive_feature],
                )
            )
        )

        status = self._status_from_metrics(metrics)
        flags = self._collect_flags(metrics)
        summary = self._build_summary(metrics, subgroup_data)
        recommendations = self._build_recommendations(metrics, subgroup_data)

        report = {
            "dataset": self.dataset,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "flags": flags,
            "summary": summary,
            "recommendations": recommendations,
            "thresholds": {
                "pass_eod": self.thresholds.pass_eod,
                "warn_eod": self.thresholds.warn_eod,
                "pass_di": self.thresholds.pass_di,
            },
            "metrics": {
                "equal_opportunity_difference": {
                    "value": metrics["equal_opportunity_difference"],
                    "ci_95": [eod_ci[0], eod_ci[1]],
                },
                "demographic_parity_difference": {
                    "value": metrics["demographic_parity_difference"],
                    "ci_95": [dpd_ci[0], dpd_ci[1]],
                },
                "equalized_odds_difference": {
                    "value": metrics["equalized_odds_difference"],
                    "ci_95": [eodds_ci[0], eodds_ci[1]],
                },
                "demographic_parity_ratio": {
                    "value": metrics["demographic_parity_ratio"],
                },
                "disparate_impact": {
                    "value": metrics["disparate_impact"],
                },
            },
            "group_metrics": subgroup_data,
            "validation": {
                "fairlearn_equal_opportunity_difference": metrics["equal_opportunity_difference"],
                "fairlearn_demographic_parity_difference": metrics["demographic_parity_difference"],
                "fairlearn_equalized_odds_difference": metrics["equalized_odds_difference"],
                "fairlearn_demographic_parity_ratio": metrics["demographic_parity_ratio"],
            },
        }
        return report

    def save_report(
        self,
        report: Dict[str, Any],
        dataset: str | None = None,
        output_dir: str = "modules/fairness/outputs/reports",
    ) -> str:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        name = dataset or self.dataset
        filename = f"fairness_report_{name}.json"
        path = out_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return str(path)