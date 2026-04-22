# Check 2: duplicate rows across splits

import pandas as pd
import src.leakage_utils as lu

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Hash every row in train-test DataFrames
train_hashes = X_train.apply(lambda x: hash(tuple(x)), axis=1)
test_hashes = X_test.apply(lambda x: hash(tuple(x)), axis=1)

# Find common hashes
duplicate_rows = set(train_hashes) & set(test_hashes)

# Compute duplicate_rows / len(df_test)
contamination = len(duplicate_rows) / len(X_test)

def assign_severity(contamination):
    if contamination > lu.LEAKAGE_THRESHOLDS["duplicate_row_ratio_medium"]:
        severity = "medium"
        if contamination > lu.LEAKAGE_THRESHOLDS["duplicate_row_ratio_high"]:
            severity = "high"
    return severity

