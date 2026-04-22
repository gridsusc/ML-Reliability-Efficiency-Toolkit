# Threshold constants and shared helpers

LEAKAGE_THRESHOLDS = {
    # Check 1 — Target Correlation
    "target_correlation_medium":  0.85,    # |r| >= this → MEDIUM
    "target_correlation_high":    0.95,    # |r| >= this → HIGH

    # Check 2 — Train/Test Contamination
    "duplicate_row_ratio_medium": 0.0,     # >0% → MEDIUM
    "duplicate_row_ratio_high":   0.01,    # ratio > 1% → HIGH

    # Check 3 — Future Information
    # MEDIUM: any future-dated value or uncertain reference date
    # HIGH:   clear systematic future-dated values
}