from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .fairness_pipeline import FairnessPipeline


def load_adult_income() -> pd.DataFrame:
    data = fetch_openml(name="adult", version=2, as_frame=True)
    df = data.frame.copy()

    df = df.replace("?", pd.NA).dropna().reset_index(drop=True)

    if "class" not in df.columns:
        raise KeyError("Expected column 'class' not found in Adult dataset.")

    df["income"] = (df["class"] == ">50K").astype(int)
    df = df.drop(columns=["class"])
    return df


def train_baseline(df: pd.DataFrame, target_col: str = "income"):
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    sensitive_cols = [c for c in ["sex", "race"] if c in X.columns]
    feature_cols = [c for c in X.columns if c not in sensitive_cols]
    X = X[feature_cols].copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=20000, solver="liblinear")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    test_df = df.loc[X_test.index].copy()
    test_df["predicted_income"] = preds

    return model, test_df, y_test


def run_fairness_audit():
    df = load_adult_income()
    _, test_df, _ = train_baseline(df)

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs" / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for sensitive_feature in ["sex", "race"]:
        if sensitive_feature not in test_df.columns:
            print(f"Skipping {sensitive_feature}: column not found.")
            continue

        dataset_name = f"adult_income_{sensitive_feature}"

        pipeline = FairnessPipeline(
            df=test_df,
            target_col="income",
            prediction_col="predicted_income",
            sensitive_feature=sensitive_feature,
            dataset=dataset_name,
            model="LogisticRegression",
        )

        report = pipeline.run_all_metrics()
        report_path = pipeline.save_report(report, dataset=dataset_name)

        eod = report["metrics"]["equal_opportunity_difference"]["value"]
        eod_ci = report["metrics"]["equal_opportunity_difference"]["ci_95"]

        print(f"\nSensitive feature: {sensitive_feature}")
        print(f"Status: {report['status']}")
        print(f"EOD: {eod:.4f} ± {((eod_ci[1] - eod_ci[0]) / 2):.4f}")
        print(f"Saved report: {report_path}")


if __name__ == "__main__":
    run_fairness_audit()