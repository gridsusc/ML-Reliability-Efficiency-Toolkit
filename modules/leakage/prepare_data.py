def load_data():
    import pandas as pd
    import numpy as np
    from ucimlrepo import fetch_ucirepo 
    from sklearn.model_selection import train_test_split
    
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 

    # The target is already separated by fetch_ucirepo
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # 1. Handle missing values
    # In the Adult dataset, missing values are sometimes represented as '?'
    X.replace('?', np.nan, inplace=True)

    # For simplicity, we drop rows with missing values
    # We need to drop the same rows from y to keep them aligned
    missing_mask = X.isnull().any(axis=1) | y.isnull().any(axis=1)
    X_clean = X[~missing_mask]
    y_clean = y[~missing_mask]

    # 2. Encode categorical columns in features
    # Use pd.get_dummies to one-hot encode categorical features
    X_encoded = pd.get_dummies(X_clean, drop_first=True)

    # 3. Label encode the target column
    # Clean string values first ('<=50K.' becomes '<=50K')
    y_clean = y_clean['income'].str.replace('.', '', regex=False)
    y_encoded = y_clean.astype('category').cat.codes

    print("Data Cleaning Complete!")
    print(f"Original shape: X={X.shape}, y={y.shape}")
    print(f"Cleaned and encoded shape: X={X_encoded.shape}, y={y_encoded.shape}")

    # 4. Stratified Train/Test Split (80/20)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, 
        y_encoded, 
        test_size=0.20, 
        stratify=y_encoded, 
        random_state=42
    )

    # save the splits to data/processed/ folder
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_data()

# The splits are kept separate for the pipeline!