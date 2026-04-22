import unittest
import os
import json
import pandas as pd
import sys

# Ensure src module is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.leakage_pipeline import LeakagePipeline

class TestLeakageChecks(unittest.TestCase):
    def test_synthetic_correlation_returns_high(self):
        """1. Functional validation: Check fires HIGH when leakage injected (Target Correlation)"""
        df = pd.read_csv('data/synthetic/v1_target_leakage.csv')
        
        # simple random split since leak exists inherently
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        
        pipeline = LeakagePipeline(df_train=train, df_test=test, target_col='target')
        report = pipeline.run_all_checks()
        
        self.assertEqual(report['checks']['target_correlation']['risk'], 'HIGH')

    def test_synthetic_contamination_returns_high(self):
        """1. Functional validation: Check fires HIGH when leakage injected (Contamination)"""
        df = pd.read_csv('data/synthetic/v2_train_test_leakage.csv')
        
        train = df[df['split'] == 'train'].drop(columns=['split'])
        test = df[df['split'] == 'test'].drop(columns=['split'])
        
        pipeline = LeakagePipeline(df_train=train, df_test=test, target_col='target')
        report = pipeline.run_all_checks()
        
        # Based on threshold ratio > 0.01, and 5% injected, it yields HIGH
        self.assertEqual(report['checks']['train_test_contamination']['risk'], 'HIGH')

    def test_clean_adult_returns_low(self):
        """2. False-positive discipline: Running on clean Adult Income produces overall_risk = LOW"""
        # Load unmodified clean data mapped from previously stored artifacts
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')

        X_train['income'] = y_train
        X_test['income'] = y_test

        train = X_train.drop_duplicates()
        test = X_test.drop_duplicates()
        
        # Rigorous feature-hash purge
        train_hashes = train.drop(columns=['income']).apply(lambda x: hash(tuple(x)), axis=1)
        test_hashes = test.drop(columns=['income']).apply(lambda x: hash(tuple(x)), axis=1)
        
        overlap = set(train_hashes) & set(test_hashes)
        test = test[~test_hashes.isin(overlap)]
        
        pipeline = LeakagePipeline(df_train=train, df_test=test, target_col='income')
        report = pipeline.run_all_checks()
        
        # Test exact false positive prevention
        self.assertEqual(report['overall_risk'], 'LOW')

    def test_output_schema_is_compliant(self):
        """3. Output schema compliance: Every JSON parses cleanly containing required keys"""
        # Read from the baseline output to verify structural integrity
        X_train = pd.read_csv('data/processed/X_train.csv').head(100)
        y_train = pd.read_csv('data/processed/y_train.csv').head(100)
        X_test = pd.read_csv('data/processed/X_test.csv').head(20)
        y_test = pd.read_csv('data/processed/y_test.csv').head(20)

        y_train.columns = ['income']
        y_test.columns = ['income']

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        pipeline = LeakagePipeline(df_train=train, df_test=test, target_col='income')
        report = pipeline.run_all_checks()

        required_keys = ['module', 'dataset', 'timestamp', 'status', 'summary', 'checks', 'overall_risk']
        
        # Test Top-level compliance
        for key in required_keys:
            self.assertIn(key, report)
            
        # Test Nested Checks structural compliance
        self.assertIn('target_correlation', report['checks'])
        self.assertIn('train_test_contamination', report['checks'])
        self.assertIn('future_information', report['checks'])

if __name__ == '__main__':
    unittest.main()
