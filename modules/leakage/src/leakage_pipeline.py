# Orchestrator: runs all checks, emits JSON report

import os
import json
import pandas as pd
from datetime import datetime
import src.leakage_utils as lu

class LeakagePipeline:
    def __init__(self, df_train, df_test, target_col, datetime_cols=None, reference_date=None, params=None):
        self.X_train = df_train.drop(columns=[target_col])
        self.y_train = df_train[target_col]
        self.X_test = df_test.drop(columns=[target_col])
        self.y_test = df_test[target_col]
        self.target_col = target_col
        self.datetime_cols = datetime_cols
        self.reference_date = reference_date
        self.params = params if params is not None else {}
        self.thresholds = lu.LEAKAGE_THRESHOLDS

    def check_target_correlation(self):
        # Pearson for numeric, Spearman for bool
        corr_num = self.X_train.select_dtypes(include=['number']).corrwith(self.y_train, method='pearson')
        corr_cat = self.X_train.select_dtypes(include=['bool']).corrwith(self.y_train, method='spearman')
        
        corr_combined = pd.concat([corr_num, corr_cat]).abs().fillna(0)
        
        max_corr = corr_combined.max() if not corr_combined.empty else 0
        flagged_features = corr_combined[corr_combined >= self.thresholds["target_correlation_medium"]].index.tolist()
        
        risk = "LOW"
        if max_corr >= self.thresholds["target_correlation_high"]:
            risk = "HIGH"
        elif max_corr >= self.thresholds["target_correlation_medium"]:
            risk = "MEDIUM"
            
        return {
            "flagged_features": flagged_features,
            "max_correlation": float(max_corr),
            "risk": risk
        }

    def check_train_test_contamination(self):
        train_hashes = self.X_train.apply(lambda x: hash(tuple(x)), axis=1)
        test_hashes = self.X_test.apply(lambda x: hash(tuple(x)), axis=1)
        
        duplicate_rows = set(train_hashes) & set(test_hashes)
        duplicates_count = len(duplicate_rows)
        contamination_ratio = duplicates_count / len(self.X_test) if len(self.X_test) > 0 else 0
        
        risk = "LOW"
        if contamination_ratio > self.thresholds["duplicate_row_ratio_high"]:
            risk = "HIGH"
        elif contamination_ratio > self.thresholds["duplicate_row_ratio_medium"]:
            risk = "MEDIUM"
            
        return {
            "duplicate_rows": duplicates_count,
            "contamination_ratio": float(contamination_ratio),
            "risk": risk
        }

    def check_future_information(self):
        if not self.datetime_cols or not self.reference_date:
            return {
                "flagged_columns": [],
                "risk": "SKIPPED"
            }
            
        date_converted = pd.to_datetime(self.reference_date)
        flagged_columns = []
        future_leakage_found = False
        
        for col in self.datetime_cols:
            if col in self.X_train.columns:
                train_dates = pd.to_datetime(self.X_train[col])
                test_dates = pd.to_datetime(self.X_test[col]) if col in self.X_test.columns else []
                
                if (train_dates > date_converted).any():
                    flagged_columns.append(col)
                    future_leakage_found = True
                elif len(test_dates) > 0 and (test_dates > date_converted).any():
                    if col not in flagged_columns:
                        flagged_columns.append(col)
                    future_leakage_found = True

        risk = "HIGH" if future_leakage_found else "PASS"
        
        return {
            "flagged_columns": flagged_columns,
            "risk": risk
        }

    def run_all_checks(self):
        corr_results = self.check_target_correlation()
        contam_results = self.check_train_test_contamination()
        future_results = self.check_future_information()
        
        risks = [corr_results["risk"], contam_results["risk"], future_results["risk"]]
        
        if "HIGH" in risks:
            overall_risk = "HIGH"
        elif "MEDIUM" in risks:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
            
        status = "FAIL" if overall_risk in ["HIGH", "MEDIUM"] else "PASS"
        
        if overall_risk == "HIGH":
            summary = "HIGH risk detected. Review flagged features and logic."
        elif overall_risk == "MEDIUM":
            summary = "MEDIUM risk detected. Review flagged features and logic."
        else:
            summary = "No significant leakage detected."
            
        report = {
            "module": "leakage",
            "dataset": self.params.get("dataset_name", "unknown_dataset"),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "summary": summary,
            "checks": {
                "target_correlation": corr_results,
                "train_test_contamination": contam_results,
                "future_information": future_results
            },
            "overall_risk": overall_risk
        }
        
        return report

    def save_report(self, report, dataset="dataset"):
        report["dataset"] = dataset
        
        output_dir = "outputs/reports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{dataset}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to {filepath}")
        return filepath
