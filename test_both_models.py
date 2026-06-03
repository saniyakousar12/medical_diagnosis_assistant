
import joblib
import numpy as np

print("="*50)
print("TESTING BOTH MODELS")
print("="*50)

# Test Heart Model
try:
    heart_model = joblib.load('modules/module2_ml/model_heart.pkl')
    print("\n✅ Heart model loaded successfully!")
    
    # Sample patient for heart disease
    sample_heart = np.array([[55, 1, 1, 130, 250, 0, 1, 150, 0, 1.0, 1, 0, 1]])
    pred = heart_model.predict(sample_heart)
    prob = heart_model.predict_proba(sample_heart)[0][1]
    print(f"   Sample prediction: {'Disease' if pred[0]==1 else 'No Disease'}")
    print(f"   Probability: {prob:.2%}")
except Exception as e:
    print(f"\n❌ Heart model error: {e}")

# Test Diabetes Model
try:
    diabetes_model = joblib.load('modules/module2_ml/model_diabetes.pkl')
    print("\n✅ Diabetes model loaded successfully!")
    
    # Sample patient for diabetes
    sample_diabetes = np.array([[2, 120, 70, 20, 80, 25.0, 0.5, 35]])
    pred = diabetes_model.predict(sample_diabetes)
    prob = diabetes_model.predict_proba(sample_diabetes)[0][1]
    print(f"   Sample prediction: {'Diabetes' if pred[0]==1 else 'No Diabetes'}")
    print(f"   Probability: {prob:.2%}")
except Exception as e:
    print(f"\n❌ Diabetes model error: {e}")

print("\n" + "="*50)
print("🎉 Module 2 Complete!")
print("="*50)