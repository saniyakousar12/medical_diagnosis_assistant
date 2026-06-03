import sys
import os
import traceback

print("="*50)
print("DIABETES MODEL DEBUG")
print("="*50)

# Test 1: Check if files exist
print("\n1. Checking files...")
heart_model = os.path.exists('modules/module2_ml/model_heart.pkl')
heart_scaler = os.path.exists('modules/module1_data/scaler_heart.pkl')
diabetes_model = os.path.exists('modules/module2_ml/model_diabetes.pkl')
diabetes_scaler = os.path.exists('modules/module1_data/scaler_diabetes.pkl')

print(f"   Heart model: {heart_model}")
print(f"   Heart scaler: {heart_scaler}")
print(f"   Diabetes model: {diabetes_model}")
print(f"   Diabetes scaler: {diabetes_scaler}")

# Test 2: Try to load diabetes model
print("\n2. Attempting to load diabetes model...")
try:
    import joblib
    model = joblib.load('modules/module2_ml/model_diabetes.pkl')
    print(f"   ✅ Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    traceback.print_exc()

# Test 3: Try to load diabetes scaler
print("\n3. Attempting to load diabetes scaler...")
try:
    scaler = joblib.load('modules/module1_data/scaler_diabetes.pkl')
    print(f"   ✅ Scaler loaded")
except Exception as e:
    print(f"   ❌ Error loading scaler: {e}")

# Test 4: Try to make a prediction
print("\n4. Attempting prediction...")
try:
    import pandas as pd
    import numpy as np
    
    # Create test patient
    test_patient = pd.DataFrame([{
        'Pregnancies': 3,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 80,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }])
    
    # Scale
    test_scaled = scaler.transform(test_patient)
    
    # Predict
    proba = model.predict_proba(test_scaled)[0][1]
    print(f"   ✅ Prediction successful!")
    print(f"   Probability: {proba:.4f}")
    print(f"   Risk: {'High' if proba > 0.6 else 'Medium' if proba > 0.3 else 'Low'}")
    
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    traceback.print_exc()