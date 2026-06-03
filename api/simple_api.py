from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback

app = FastAPI()

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def root():
    return {"message": "API is running", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "server": "running"}

@app.post("/predict/heart")
def predict_heart(patient: HeartInput):
    try:
        # Simple mock prediction based on age and cholesterol
        risk = 0.2
        
        if patient.age > 50:
            risk += 0.2
        if patient.chol > 240:
            risk += 0.3
        if patient.trestbps > 140:
            risk += 0.2
        if patient.thalach < 120:
            risk += 0.1
            
        probability = min(0.95, risk)
        
        if probability < 0.3:
            risk_level = "Low Risk"
            color = "green"
            advice = "✅ Maintain healthy lifestyle"
        elif probability < 0.6:
            risk_level = "Medium Risk"
            color = "orange"
            advice = "⚠️ Consult a doctor"
        else:
            risk_level = "High Risk"
            color = "red"
            advice = "🚨 Seek medical attention"
        
        return {
            'prediction': 1 if probability > 0.5 else 0,
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': color,
            'risk_advice': advice,
            'shap_waterfall': '',
            'explanations': [
                f"Age: {patient.age} years",
                f"Cholesterol: {patient.chol} mg/dL",
                f"Blood pressure: {patient.trestbps} mmHg"
            ]
        }
    except Exception as e:
        print(f"Heart endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/diabetes")
def predict_diabetes(patient: DiabetesInput):
    try:
        print(f"Received diabetes data: {patient.dict()}")  # Debug print
        
        # Simple mock calculation
        risk = 0.1
        
        # Glucose risk
        if patient.Glucose > 140:
            risk += 0.4
        elif patient.Glucose > 120:
            risk += 0.2
            
        # BMI risk
        if patient.BMI > 30:
            risk += 0.3
        elif patient.BMI > 25:
            risk += 0.1
            
        # Age risk
        if patient.Age > 50:
            risk += 0.1
            
        # Blood pressure risk
        if patient.BloodPressure > 90:
            risk += 0.1
            
        # Pregnancy risk
        if patient.Pregnancies > 5:
            risk += 0.1
            
        probability = min(0.95, risk)
        
        if probability < 0.3:
            risk_level = "Low Risk"
            color = "green"
            advice = "✅ Maintain healthy lifestyle. Regular check-ups recommended."
        elif probability < 0.6:
            risk_level = "Medium Risk"
            color = "orange"
            advice = "⚠️ Monitor glucose levels. Consult a doctor."
        else:
            risk_level = "High Risk"
            color = "red"
            advice = "🚨 Seek medical attention immediately."
        
        result = {
            'prediction': 1 if probability > 0.5 else 0,
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': color,
            'risk_advice': advice,
            'shap_waterfall': '',
            'explanations': [
                f"Glucose level: {patient.Glucose} mg/dL",
                f"BMI: {patient.BMI}",
                f"Age: {patient.Age} years",
                f"Blood pressure: {patient.BloodPressure} mmHg"
            ]
        }
        
        print(f"Returning result: {result}")  # Debug print
        return result
        
    except Exception as e:
        print(f"Diabetes endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API server on http://127.0.0.1:8000")
    print("📊 Heart endpoint: POST /predict/heart")
    print("📊 Diabetes endpoint: POST /predict/diabetes")
    uvicorn.run(app, host="127.0.0.1", port=8000)