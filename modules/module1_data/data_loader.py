import pandas as pd
import os
import numpy as np

def load_heart():
    """Load UCI Heart Disease dataset - Properly handles raw Cleveland data"""
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, 'data', 'heart.csv')
    
    # Load raw data (UCI Cleveland dataset has no headers)
    # The dataset has 303 rows and 14 columns (13 features + 1 target)
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        # Try reading with header=None first (raw UCI format)
        df = pd.read_csv(file_path, header=None, names=column_names)
    except:
        # If that fails, try reading as regular CSV with header
        df = pd.read_csv(file_path)
        if len(df.columns) == 14:
            df.columns = column_names
        elif 'target' not in df.columns:
            # Assume last column is target
            target_col = df.columns[-1]
            df.rename(columns={target_col: 'target'}, inplace=True)
    
    # Clean the data
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Convert ca and thal to numeric (they might be objects)
    if 'ca' in df.columns:
        df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    if 'thal' in df.columns:
        df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    # Drop rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with missing values")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Convert target to binary (0 = no disease, 1 = disease)
    # Original: 0 = healthy, 1,2,3,4 = disease levels
    y = (y > 0).astype(int)
    
    print(f"Heart dataset loaded: {len(X)} rows, {len(X.columns)} features")
    print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    if len(X) < 100:
        print(f"⚠️ WARNING: Only {len(X)} rows loaded. Expected 303 rows.")
        print("Please check your heart.csv file.")
    
    return X, y

def load_diabetes():
    """Load Pima Indians Diabetes dataset"""
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, 'data', 'diabetes.csv')
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check if 'Outcome' column exists, if not, assume last column is target
    if 'Outcome' in df.columns:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
    else:
        # Last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    print(f"Diabetes dataset loaded: {len(X)} rows, {len(X.columns)} features")
    print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    return X, y

# Test the function
if __name__ == "__main__":
    print("Testing data loaders...")
    X_h, y_h = load_heart()
    print(f"Heart: X shape={X_h.shape}, y shape={y_h.shape}")
    
    X_d, y_d = load_diabetes()
    print(f"Diabetes: X shape={X_d.shape}, y shape={y_d.shape}")
    
    print("\n✅ Data loaders ready!")