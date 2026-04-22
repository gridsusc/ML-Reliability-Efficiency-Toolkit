# Check 1: features with strong target association

import src.leakage_utils as lu
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

# Correlation check
# Pearson correlation for numeric features
corr_num = X_train[X_train.select_dtypes(include=['number']).columns].corrwith(y_train, method='pearson')

# Spearman correlation for categorical features
corr_cat = X_train[X_train.select_dtypes(include=['bool']).columns].corrwith(y_train, method='spearman')

# Combine numeric and bool correlations into one Series
corr_combined = pd.concat([corr_num, corr_cat])

# Take the absolue value
corr_abs = corr_combined.abs()

# Severity classification logic
def assign_severity(corr_value):
    if corr_value >= lu.LEAKAGE_THRESHOLDS["target_correlation_high"]:
        return "high"
    elif corr_value >= lu.LEAKAGE_THRESHOLDS["target_correlation_medium"]:
        return "medium"
    else:
        return "low"

severity_level = corr_abs.apply(assign_severity)

# Combine raw correlation and severity into a single report DataFrame
report_df = pd.DataFrame({
    'Raw Correlation': corr_combined,
    'Severity': severity_level
})

# Sort by absolute correlation to bubble the highest risks to the top
report_df['Absolute Correlation'] = corr_abs
report_df = report_df.sort_values(by='Absolute Correlation', ascending=False)
report_df = report_df.drop(columns=['Absolute Correlation'])

print("\nTarget Correlation Check:")
print(report_df.head(20)) # Print the top 20 most correlated features
