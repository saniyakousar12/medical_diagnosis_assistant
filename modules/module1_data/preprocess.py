import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now imports will work
from modules.module1_data.data_loader import load_heart, load_diabetes
from modules.module1_data.preprocess import preprocess_data

def train_and_save_models():
    """Train models for both datasets"""
    
    # Create models directory
    os.makedirs('modules/module2_ml', exist_ok=True)
    
    # Process Heart Dataset
    print("\n" + "="*60)
    print("HEART DISEASE MODEL")
    print("="*60)
    
    X_heart, y_heart = load_heart()
    X_train_h, X_test_h, y_train_h, y_test_h, _ = preprocess_data(X_heart, y_heart, 'heart')
    
    # Train models for heart
    train_dataset_models(X_train_h, X_test_h, y_train_h, y_test_h, 'heart')
    
    # Process Diabetes Dataset
    print("\n" + "="*60)
    print("DIABETES MODEL")
    print("="*60)
    
    X_diab, y_diab = load_diabetes()
    X_train_d, X_test_d, y_train_d, y_test_d, _ = preprocess_data(X_diab, y_diab, 'diabetes')
    
    # Train models for diabetes
    train_dataset_models(X_train_d, X_test_d, y_train_d, y_test_d, 'diabetes')
    
    print("\n" + "="*60)
    print("✅ MODULE 2 COMPLETED SUCCESSFULLY!")
    print("="*60)

def train_dataset_models(X_train, X_test, y_train, y_test, dataset_name):
    """Train multiple models and return best one"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Other metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'test_auc': float(test_auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"   Test AUC: {test_auc:.4f}, Accuracy: {accuracy:.4f}")
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
    
    # Hyperparameter tuning for XGBoost
    print(f"\n🔧 Tuning XGBoost...")
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    random_search = RandomizedSearchCV(xgb, param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    best_xgb = random_search.best_estimator_
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, y_pred_proba)
    
    results['XGBoost (Tuned)'] = {
        'best_params': random_search.best_params_,
        'test_auc': float(tuned_auc),
        'accuracy': float(accuracy_score(y_test, best_xgb.predict(X_test)))
    }
    
    print(f"   Tuned XGBoost AUC: {tuned_auc:.4f}")
    
    if tuned_auc > best_auc:
        best_model = best_xgb
        best_auc = tuned_auc
    
    # Save results
    with open(f'modules/module2_ml/results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best model
    joblib.dump(best_model, f'modules/module2_ml/model_{dataset_name}.pkl')
    print(f"\n✅ Best model saved for {dataset_name} with AUC: {best_auc:.4f}")
    
    return best_model

if __name__ == "__main__":
    train_and_save_models()