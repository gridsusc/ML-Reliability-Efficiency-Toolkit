# Fairness Module

Evaluates model behavior across demographic subgroups and computes fairness metrics. 
Designed to surface bias, error asymmetries, and performance disparities in classification models, 
using the Adult Income and German Credit datasets as primary benchmarks.

---

## Files

---

## How to Use



---

## Metrics

This module computes multiple complementary fairness metrics. 
No single metric is sufficient — all must be reported together.

---

### 1. Equal Opportunity Difference (Primary)

### 2. Demographic Parity Ratio (Secondary)

### 3. Accuracy Difference Across Groups (Tertiary)

### 4. Error Rate Disparity (Critical)

---

## Base Rate Awareness

The module reports base rates per group:

```
Base rates (P(Y=1)):
  Male   : 0.42
  Female : 0.28
```

**Why this matters:**  
Differences in base rates explain why fairness metrics can conflict. 
Demographic parity may be inappropriate when base rates differ.

---

## Statistical Confidence

All reported fairness metrics include uncertainty estimates:

```
Equal Opportunity Difference: 0.14 ± 0.03
```

Confidence is estimated using bootstrap resampling.

---

## Direction of Harm (Interpretability Layer)

Each metric includes a human-readable explanation:

```
Equal Opportunity Difference: 0.14
→ Model favors Male group in correctly identifying positives
→ Female group is under-selected despite qualification
```

This translates metrics into real-world impact.

---

## Metric Selection Rationale

Equal Opportunity is prioritized because:

- Task is merit-based (income / credit risk)
- Focus is fairness among qualified individuals
- Demographic parity is not appropriate when base rates differ

Other metrics are reported for context, not optimization.

---

## Example Output

```
Sensitive attribute : gender
Groups detected     : ['Male', 'Female']

Base rates:
  Male   : 0.42
  Female : 0.28

Equal Opportunity Difference  :  0.14 ± 0.03   ← FLAG
Demographic Parity Ratio      :  0.73           ← Borderline
Accuracy Difference           :  0.06           ← Investigate

FPR Difference                :  0.12           ← FLAG
FNR Difference                :  0.09           ← Investigate

Interpretation:
→ Model favors Male group in positive classification
→ Female group experiences under-selection despite qualification
→ Elevated FPR indicates potential over-penalization of one group

Per-group accuracy:
  Male   : 0.87
  Female : 0.81

Per-group TPR:
  Male   : 0.79
  Female : 0.65
```

---

## Important 

**Metric incompatibility:**  
Fairness definitions conflict. Do not optimize for all simultaneously.

**Dataset scope:**  
Adult Income and German Credit are outdated and context-specific.

**No metric proves fairness:**  
Metrics surface disparities. Interpretation requires domain judgment.

---

## References

