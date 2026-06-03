import joblib
import pandas as pd

# Load model
try:
    model = joblib.load('modules/module2_ml/model_diabetes.pkl')
    print("✅ Diabetes model loaded")
    print(f"Model type: {type(model).__name__}")
    
    # Check if model has feature_importances_
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances: {model.feature_importances_}")
    
    # Load scaler
    scaler = joblib.load('modules/module1_data/scaler_diabetes.pkl')
    print("✅ Scaler loaded")
    
    # Expected features (for Pima dataset)
    expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    print(f"\nExpected features: {expected_features}")
    
except Exception as e:
    print(f"❌ Error: {e}")