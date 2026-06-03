import requests
import json

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

print("Testing diabetes endpoint...")
try:
    response = requests.post("http://localhost:8000/predict/diabetes", json=diabetes_data)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error response: {response.text}")
    else:
        print(f"Success: {response.json()}")
except Exception as e:
    print(f"Exception: {e}")