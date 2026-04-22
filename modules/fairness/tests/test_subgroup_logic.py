from modules.fairness.src.subgroup_analysis import compute_subgroup_metrics


def test_subgroup_metrics_keys():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    sensitive = ["M", "M", "F", "F"]
    result = compute_subgroup_metrics(y_true, y_pred, sensitive)
    assert set(result.keys()) == {"subgroup_accuracy", "subgroup_tpr", "subgroup_fpr", "subgroup_fnr", "support"}
