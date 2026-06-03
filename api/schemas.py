from pydantic import BaseModel
from typing import List, Optional, Dict

class PatientInput(BaseModel):
    # Heart disease features
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
    
    # For diabetes, we'll use a subset
    # But for simplicity, we'll create separate endpoints

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class TabularResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    risk_color: str
    risk_advice: str
    shap_waterfall: str  # base64 image
    lime_weights: Dict[str, float]
    explanations: List[str]

class XrayResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap: str  # base64 image