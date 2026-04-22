import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown cell 1
text1 = """# Leakage Module Validation & Demo
This notebook demonstrates the ML Reliability & Efficiency Toolkit's **Leakage Pipeline**. 
It runs the diagnostic pipeline across three scenarios:
1. **Clean Adult Income Dataset**: Baseline evaluation (Should ideally flag very little or no leakage).
2. **Target Correlation**: Synthetic dataset simulating target feature leakage.
3. **Train-Test Contamination**: Synthetic dataset simulating data leakage spanning across dataset splits.
"""

# Code cell 1
code1 = """import pandas as pd
import sys
import os

# Ensure the root directory is accessible so we can import the src module
sys.path.append(os.path.abspath('..'))

from src.leakage_pipeline import LeakagePipeline
"""

# Markdown cell 2
text2 = """## 1. Clean Adult Income Dataset
First, we run the diagnostic checks against the standard pre-processed Adult dataset. We drop natural duplicate rows from the dataset (which represent statistically identical individuals in the consensus) to ensure a perfectly clean baseline without accidental collision noise.
"""

# Code cell 2
code2 = """# Load the pre-processed Adult Income splits
X_train = pd.read_csv('../data/processed/X_train.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

# Concat to pass to pipeline
y_train.columns = ["income"]
y_test.columns = ["income"]
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Drop any naturally identical rows within each split
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

# Feature-hash purge: remove any test rows whose feature fingerprint
# already exists in train (identical feature rows, ignoring target).
# Uses the same hashing method as the pipeline's contamination check for consistency.
train_hashes = train_df.drop(columns=["income"]).apply(lambda x: hash(tuple(x)), axis=1)
test_hashes  = test_df.drop(columns=["income"]).apply(lambda x: hash(tuple(x)), axis=1)
overlap_hashes = set(train_hashes) & set(test_hashes)
test_df = test_df[~test_hashes.isin(overlap_hashes)]

pipeline_clean = LeakagePipeline(
    df_train=train_df,
    df_test=test_df,
    target_col="income"
)

report_clean = pipeline_clean.run_all_checks()
pipeline_clean.save_report(report_clean, dataset="adult_income_clean")

print("Clean Dataset Validation")
print(f"Overall Risk: {report_clean['overall_risk']}")
for test, result in report_clean['checks'].items():
    print(f" - {test}: {result['risk']}")
"""

# Markdown cell 3
text3 = """## 2. Synthetic Scenario: Target Correlation Leakage
In this synthetic scenario, we evaluate `v1_target_leakage.csv` which intentionally contains a column literally called `noisy_target_copy` reflecting massive information overlap with the `target`."""

# Code cell 3
code3 = """from sklearn.model_selection import train_test_split

v1_df = pd.read_csv('../data/synthetic/v1_target_leakage.csv')

# Split it
df1_train, df1_test = train_test_split(v1_df, test_size=0.2, random_state=42)

pipeline_v1 = LeakagePipeline(
    df_train=df1_train,
    df_test=df1_test,
    target_col="target"
)

report_v1 = pipeline_v1.run_all_checks()
pipeline_v1.save_report(report_v1, dataset="synthetic_v1_target")

print("V1 Validation (Target Leakage expected HIGH)")
print(f"Overall Risk: {report_v1['overall_risk']}")
for test, result in report_v1['checks'].items():
    print(f" - {test}: {result['risk']}")
"""

# Markdown cell 4
text4 = """## 3. Synthetic Scenario: Train-Test Contamination
This dataset `v2_train_test_leakage.csv` natively encodes rows overlapping across the predefined train and test split assignments."""

# Code cell 4
code4 = """v2_df = pd.read_csv('../data/synthetic/v2_train_test_leakage.csv')

df2_train = v2_df[v2_df['split'] == 'train'].drop(columns=['split'])
df2_test = v2_df[v2_df['split'] == 'test'].drop(columns=['split'])

pipeline_v2 = LeakagePipeline(
    df_train=df2_train,
    df_test=df2_test,
    target_col="target"
)

report_v2 = pipeline_v2.run_all_checks()
pipeline_v2.save_report(report_v2, dataset="synthetic_v2_contamination")

print("V2 Validation (Contamination expected HIGH/MEDIUM based on configured thresholds)")
print(f"Overall Risk: {report_v2['overall_risk']}")
for test, result in report_v2['checks'].items():
    print(f" - {test}: {result['risk']}")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

os.makedirs("notebooks", exist_ok=True)
with open('notebooks/leakage_demo.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully at notebooks/leakage_demo.ipynb")
