import os
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"
import pandas as pd
from src.leakage_pipeline import LeakagePipeline

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_train.columns = ["income"]
train_df = pd.concat([X_train, y_train], axis=1)

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
y_test.columns = ["income"]
test_df = pd.concat([X_test, y_test], axis=1)

pipeline = LeakagePipeline(
    df_train=train_df,
    df_test=test_df,
    target_col="income",
    datetime_cols=None,                     # omit if no temporal columns in Adult Income
    reference_date=None,                    # omitted
)

report = pipeline.run_all_checks()
pipeline.save_report(report, dataset="adult_income")
print("Overall Risk:", report["overall_risk"])