from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.module3_xai.explainer_simple import predict_and_explain

app = FastAPI(title="AI Medical Diagnosis Assistant")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    risk_color: str
    risk_advice: str
    shap_waterfall: str
    explanations: List[str]

@app.get("/")
def root():
    return {"message": "AI Medical Diagnosis Assistant API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict/heart", response_model=PredictionResponse)
async def predict_heart(patient: HeartInput):
    result = predict_and_explain(patient.dict(), 'heart')
    return result

@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(patient: DiabetesInput):
    result = predict_and_explain(patient.dict(), 'diabetes')
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)