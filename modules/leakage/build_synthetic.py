import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_synthetic_datasets():
    # 1. Create the output directory
    output_dir = 'data/synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Base dataset generation
    np.random.seed(42)
    n_samples = 1000
    
    # Define a base dataframe with random features
    base_df = pd.DataFrame({
        'feature_numeric_1': np.random.randn(n_samples),
        'feature_numeric_2': np.random.rand(n_samples) * 100,
        'feature_category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })

    
    # VERSION 1: Target Leakage (Noisy copy of target label)
    df_v1 = base_df.copy()
    
    # Introduce 'noisy_target_copy': 99% of the time it equals 'target', 1% random flip
    noise_mask = np.random.rand(len(df_v1)) > 0.99
    df_v1['noisy_target_copy'] = np.where(noise_mask, 1 - df_v1['target'], df_v1['target'])
    
    df_v1.to_csv(os.path.join(output_dir, 'v1_target_leakage.csv'), index=False)
    print(f"Created: {output_dir}/v1_target_leakage.csv")
    
    # VERSION 2: Train/Test Contamination (5% test duplicates)
    df_v2 = base_df.copy()
    
    # Split into train/test manually (80/20)
    train_size = int(0.8 * n_samples)
    df_train = df_v2.iloc[:train_size].copy()
    df_test = df_v2.iloc[train_size:].copy()
    
    # We will duplicate 5% of the test rows from the train set
    num_duplicates = int(0.05 * len(df_test))
    
    # Select random rows from train set to duplicate
    duplicate_indices = np.random.choice(df_train.index, num_duplicates, replace=False)
    
    # Select random rows in test set to overwrite
    test_indices_to_replace = np.random.choice(df_test.index, num_duplicates, replace=False)
    
    # Perform the exact replacement
    df_test.loc[test_indices_to_replace] = df_train.loc[duplicate_indices].values
    
    # Add a split column to formally separate them after combining
    df_train['split'] = 'train'
    df_test['split'] = 'test'
    
    df_v2_combined = pd.concat([df_train, df_test]).reset_index(drop=True)
    df_v2_combined.to_csv(os.path.join(output_dir, 'v2_train_test_leakage.csv'), index=False)
    print(f"Created: {output_dir}/v2_train_test_leakage.csv")


    # VERSION 3: Time Leakage (Dates past reference date)
    df_v3 = base_df.copy()
    
    # Assume reference date is 2024-01-01. Most data should be STRICTLY BEFORE this date.
    reference_date = datetime(2024, 1, 1)
    
    # Generate dates strictly before the reference date
    dates = [reference_date - timedelta(days=np.random.randint(1, 365)) for _ in range(n_samples)]
    
    # Select a few random rows to purposely go PAST the reference date (e.g. 50 rows)
    future_indices = np.random.choice(n_samples, 50, replace=False)
    for idx in future_indices:
        # These dates will occur 1 to 30 days AFTER the reference date
        dates[idx] = reference_date + timedelta(days=np.random.randint(1, 30))
        
    df_v3['event_date'] = dates
    df_v3.to_csv(os.path.join(output_dir, 'v3_temporal_leakage.csv'), index=False)
    print(f"Created: {output_dir}/v3_temporal_leakage.csv")

if __name__ == '__main__':
    create_synthetic_datasets()