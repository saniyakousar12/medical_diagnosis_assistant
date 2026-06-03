import requests
import json

print("="*50)
print("TESTING API ENDPOINTS")
print("="*50)

# Test health endpoint
print("\n1. Testing health endpoint...")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test heart endpoint
print("\n2. Testing heart endpoint...")
heart_data = {
    "age": 55, "sex": 1, "cp": 1, "trestbps": 130,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 1
}

try:
    response = requests.post("http://localhost:8000/predict/heart", json=heart_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Risk: {result.get('risk_level', 'N/A')}")
        print(f"   Probability: {result.get('probability', 0):.2%}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test diabetes endpoint
print("\n3. Testing diabetes endpoint...")
diabetes_data = {
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
}

try:
    response = requests.post("http://localhost:8000/predict/diabetes", json=diabetes_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Risk: {result.get('risk_level', 'N/A')}")
        print(f"   Probability: {result.get('probability', 0):.2%}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")