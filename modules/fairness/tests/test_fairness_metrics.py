from modules.fairness.src.fairness_metrics import (
    accuracy_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    fpr_difference,
    fnr_difference,
)


def test_smoke_metrics():
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0]
    sensitive = ["A", "A", "A", "B", "B", "B", "B", "A"]

    for fn in [equal_opportunity_difference, demographic_parity_ratio, accuracy_difference, fpr_difference, fnr_difference]:
        result = fn(y_true, y_pred, sensitive, random_state=0)
        assert "value" in result and "status" in result and "ci" in result
