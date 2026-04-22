# Check 3: temporal features postdating prediction time

import pandas as pd

# Accept a list of datetime column names and a reference_date
def postdate(column_names=None, reference_date=None): 
    # If parameters not provided, report as SKIPPED
    if not column_names or not reference_date:
        return "SKIPPED"

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # Convert reference_date to datetime
    date_converted = pd.to_datetime(reference_date)
    
    future_leakage_found = False
    
    # Iterate over the column names
    for col in column_names:
        if col in X_train.columns:
            # Convert both sets to datetime
            X_train[col] = pd.to_datetime(X_train[col])
            X_test[col] = pd.to_datetime(X_test[col])
            
            # Check if any values postdate the reference date in train or test
            if (X_train[col] > date_converted).any():
                print(f"[!] Future data leakage found in X_train column: '{col}'")
                future_leakage_found = True
                
            if (X_test[col] > date_converted).any():
                print(f"[!] Future data leakage found in X_test column: '{col}'")
                future_leakage_found = True

    if future_leakage_found:
        return "HIGH" # Found future data
    return "PASS"

if __name__ == '__main__':
    # Test our function without arguments to ensure it skips
    result = postdate()
    print(f"Check 3 Status: {result}")