import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

def load_heart():
    """Load UCI Heart Disease dataset"""
    file_path = 'data/heart.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        print("Please ensure heart.csv is in the data folder.")
        return None, None
    
    # Column names for heart dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load data
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Convert to numeric
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Convert to binary (0 = no disease, 1 = disease)
    y = (y > 0).astype(int)
    
    print(f"✅ Heart dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"   Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    return X, y
def load_diabetes():
    """Load Pima Indians Diabetes dataset"""
    file_path = 'data/diabetes.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return None, None
    
    df = pd.read_csv(file_path)
    
    print(f"Columns found: {df.columns.tolist()}")
    
    # Try different possible target column names
    target_column = None
    possible_targets = ['Outcome', 'outcome', 'class', 'target', 'diabetes']
    
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        # If no target column found, assume last column is target
        print(f"⚠️ No standard target column found. Using last column as target.")
        target_column = df.columns[-1]
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"✅ Diabetes dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"   Target column: '{target_column}'")
    print(f"   Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    return X, y

# ============================================
# PREPROCESSING FUNCTION
# ============================================

def preprocess_data(X, y, dataset_name):
    """Preprocess data: split, balance, scale"""
    
    print(f"\n📊 Preprocessing {dataset_name}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Apply SMOTE
    print("   Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE: {X_train.shape[0]} samples")
    
    # Scale features
    print("   Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('modules/module1_data', exist_ok=True)
    joblib.dump(scaler, f'modules/module1_data/scaler_{dataset_name}.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================
# MODEL TRAINING FUNCTION
# ============================================

def train_models_for_dataset(X_train, X_test, y_train, y_test, dataset_name):
    """Train multiple models for a dataset"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"      CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"      Test AUC: {test_auc:.4f}, Accuracy: {accuracy:.4f}")
        
        results[name] = {
            'cv_auc_mean': float(cv_scores.mean()),
            'test_auc': float(test_auc),
            'accuracy': float(accuracy)
        }
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
    
    # Tune XGBoost
    print(f"\n   🔧 Tuning XGBoost...")
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    random_search = RandomizedSearchCV(xgb, param_dist, n_iter=5, cv=3, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)
    
    best_xgb = random_search.best_estimator_
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, y_pred_proba)
    
    results['XGBoost (Tuned)'] = {
        'best_params': random_search.best_params_,
        'test_auc': float(tuned_auc)
    }
    
    print(f"      Tuned XGBoost AUC: {tuned_auc:.4f}")
    
    if tuned_auc > best_auc:
        best_model = best_xgb
        best_auc = tuned_auc
    
    # Save results and model
    os.makedirs('modules/module2_ml', exist_ok=True)
    
    with open(f'modules/module2_ml/results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    joblib.dump(best_model, f'modules/module2_ml/model_{dataset_name}.pkl')
    print(f"\n   ✅ Best model AUC: {best_auc:.4f}")
    
    return best_model

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("="*60)
    print("🚀 MODULE 2: TRAINING ML MODELS")
    print("="*60)
    
    # Create necessary directories
    os.makedirs('modules/module1_data', exist_ok=True)
    os.makedirs('modules/module2_ml', exist_ok=True)
    
    # Load Heart dataset
    print("\n📁 Loading Heart dataset...")
    X_heart, y_heart = load_heart()
    if X_heart is None:
        print("❌ Cannot proceed without heart dataset!")
        return
    
    # Preprocess Heart
    X_train_h, X_test_h, y_train_h, y_test_h = preprocess_data(X_heart, y_heart, 'heart')
    
    # Train Heart models
    print("\n" + "="*40)
    print("🎯 HEART DISEASE MODELS")
    print("="*40)
    train_models_for_dataset(X_train_h, X_test_h, y_train_h, y_test_h, 'heart')
    
    # Load Diabetes dataset
    print("\n📁 Loading Diabetes dataset...")
    X_diab, y_diab = load_diabetes()
    if X_diab is None:
        print("❌ Cannot proceed without diabetes dataset!")
        return
    
    # Preprocess Diabetes
    X_train_d, X_test_d, y_train_d, y_test_d = preprocess_data(X_diab, y_diab, 'diabetes')
    
    # Train Diabetes models
    print("\n" + "="*40)
    print("🎯 DIABETES MODELS")
    print("="*40)
    train_models_for_dataset(X_train_d, X_test_d, y_train_d, y_test_d, 'diabetes')
    
    print("\n" + "="*60)
    print("✅ MODULE 2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Models saved in: modules/module2_ml/")
    print("   - model_heart.pkl")
    print("   - model_diabetes.pkl")
    print("   - results_heart.json")
    print("   - results_diabetes.json")

if __name__ == "__main__":
    main()