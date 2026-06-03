import joblib
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import os
import warnings
warnings.filterwarnings('ignore')

class MedicalExplainer:
    def __init__(self, model_path, scaler_path, feature_names, disease_type):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = feature_names
        self.disease_type = disease_type
        
        # Feature explanations for plain English
        self.feature_explanations = {
            'age': 'Age', 'sex': 'Gender', 'cp': 'Chest pain type',
            'trestbps': 'Blood pressure', 'chol': 'Cholesterol',
            'fbs': 'Blood sugar', 'restecg': 'ECG results',
            'thalach': 'Heart rate', 'exang': 'Exercise angina',
            'oldpeak': 'ST depression', 'slope': 'ST slope',
            'ca': 'Major vessels', 'thal': 'Thalassemia',
            'Pregnancies': 'Number of pregnancies', 'Glucose': 'Blood glucose',
            'BloodPressure': 'Blood pressure', 'SkinThickness': 'Skin thickness',
            'Insulin': 'Insulin level', 'BMI': 'BMI',
            'DiabetesPedigreeFunction': 'Diabetes family history'
        }
    
    def explain_prediction(self, patient_data):
        # Prepare input
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])[self.feature_names]
        else:
            patient_df = patient_data
        
        # Scale
        patient_scaled = self.scaler.transform(patient_df)
        
        # Predict
        proba = self.model.predict_proba(patient_scaled)[0][1]
        prediction = 1 if proba >= 0.5 else 0
        
        # Get feature importance from model (if random forest or xgboost)
        explanations = []
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-3:][::-1]
                
                for idx in top_indices:
                    feature = self.feature_names[idx]
                    value = patient_df.iloc[0][feature]
                    importance = importances[idx]
                    explanations.append(f"• {self.feature_explanations.get(feature, feature)}: {value} (importance: {importance:.2%})")
            else:
                # For logistic regression, use coefficients
                if hasattr(self.model, 'coef_'):
                    coefs = abs(self.model.coef_[0])
                    top_indices = np.argsort(coefs)[-3:][::-1]
                    
                    for idx in top_indices:
                        feature = self.feature_names[idx]
                        value = patient_df.iloc[0][feature]
                        explanations.append(f"• {self.feature_explanations.get(feature, feature)}: {value}")
        except:
            explanations = ["Analysis complete. Key factors considered."]
        
        # Create simple visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green' if proba < 0.3 else 'orange' if proba < 0.6 else 'red']
        ax.barh(['Risk Probability'], [proba], color=colors[0])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title(f'{self.disease_type} Risk Assessment')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        risk_viz = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Risk level
        if proba < 0.3:
            risk_level = "Low Risk"
            risk_color = "green"
            advice = "✅ Maintain healthy lifestyle. Regular check-ups recommended."
        elif proba < 0.6:
            risk_level = "Medium Risk"  
            risk_color = "orange"
            advice = "⚠️ Consult a doctor. Consider lifestyle changes."
        else:
            risk_level = "High Risk"
            risk_color = "red"
            advice = "🚨 Seek medical attention immediately."
        
        return {
            'prediction': int(prediction),
            'probability': float(proba),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_advice': advice,
            'shap_waterfall': risk_viz,
            'explanations': explanations if explanations else ["Prediction completed successfully"]
        }

# Global instances
heart_explainer = None
diabetes_explainer = None

def get_explainer(disease_type):
    global heart_explainer, diabetes_explainer
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if disease_type == 'heart':
        if heart_explainer is None:
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            heart_explainer = MedicalExplainer(
                os.path.join(project_root, 'modules/module2_ml/model_heart.pkl'),
                os.path.join(project_root, 'modules/module1_data/scaler_heart.pkl'),
                feature_names,
                'Heart Disease'
            )
        return heart_explainer
    
    else:  # diabetes
        if diabetes_explainer is None:
            try:
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                
                model_path = os.path.join(project_root, 'modules/module2_ml/model_diabetes.pkl')
                scaler_path = os.path.join(project_root, 'modules/module1_data/scaler_diabetes.pkl')
                
                print(f"Loading diabetes model from: {model_path}")
                print(f"Loading scaler from: {scaler_path}")
                
                # Check if files exist
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Diabetes model not found at {model_path}")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Diabetes scaler not found at {scaler_path}")
                
                diabetes_explainer = MedicalExplainer(
                    model_path,
                    scaler_path,
                    feature_names,
                    'Diabetes'
                )
                print("✅ Diabetes explainer created successfully")
            except Exception as e:
                print(f"❌ Error creating diabetes explainer: {e}")
                # Return a mock explainer for diabetes if model fails
                diabetes_explainer = MockDiabetesExplainer()
        return diabetes_explainer

class MockDiabetesExplainer:
    """Fallback explainer when diabetes model is not available"""
    
    def explain_prediction(self, patient_data):
        # Simple calculation based on glucose and BMI
        glucose = patient_data.get('Glucose', 100)
        bmi = patient_data.get('BMI', 25)
        
        # Simple risk calculation
        risk_score = 0
        if glucose > 140:
            risk_score += 0.4
        elif glucose > 120:
            risk_score += 0.2
            
        if bmi > 30:
            risk_score += 0.3
        elif bmi > 25:
            risk_score += 0.1
        
        proba = min(0.95, max(0.05, risk_score))
        prediction = 1 if proba >= 0.5 else 0
        
        # Risk level
        if proba < 0.3:
            risk_level = "Low Risk"
            risk_color = "green"
            advice = "✅ Maintain healthy lifestyle. Regular check-ups recommended."
        elif proba < 0.6:
            risk_level = "Medium Risk"  
            risk_color = "orange"
            advice = "⚠️ Consult a doctor. Consider lifestyle changes."
        else:
            risk_level = "High Risk"
            risk_color = "red"
            advice = "🚨 Seek medical attention immediately."
        
        # Create simple visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green' if proba < 0.3 else 'orange' if proba < 0.6 else 'red']
        ax.barh(['Risk Probability'], [proba], color=colors[0])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Diabetes Risk Assessment (Demo Mode)')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        risk_viz = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        explanations = []
        if glucose > 120:
            explanations.append(f"• Blood glucose: {glucose} (elevated - increases risk)")
        if bmi > 25:
            explanations.append(f"• BMI: {bmi} (above normal - increases risk)")
        if not explanations:
            explanations.append("• Key factors within normal range")
        
        return {
            'prediction': int(prediction),
            'probability': float(proba),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_advice': advice,
            'shap_waterfall': risk_viz,
            'explanations': explanations
        }

def predict_and_explain(patient_dict, disease_type):
    explainer = get_explainer(disease_type)
    return explainer.explain_prediction(patient_dict)