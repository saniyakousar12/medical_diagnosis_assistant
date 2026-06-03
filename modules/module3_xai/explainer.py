import joblib
import numpy as np
import pandas as pd
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt
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
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
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
        
        # SHAP explanation
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(patient_scaled)
            
            # Create SHAP waterfall plot as base64
            fig, ax = plt.subplots(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]
                base_value = explainer.expected_value[1]
            else:
                shap_values_to_plot = shap_values
                base_value = explainer.expected_value
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_to_plot[0],
                    base_values=base_value,
                    data=patient_df.values[0],
                    feature_names=self.feature_names
                ),
                show=False,
                max_display=10
            )
            plt.title(f'{self.disease_type} Risk Assessment')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            shap_image = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # Get top features
            shap_importance = np.abs(shap_values_to_plot[0])
            top_indices = np.argsort(shap_importance)[-3:][::-1]
            
            explanations = []
            for idx in top_indices:
                if shap_importance[idx] > 0.01:
                    feature = self.feature_names[idx]
                    value = patient_df.iloc[0][feature]
                    impact = shap_values_to_plot[0][idx]
                    direction = "increased" if impact > 0 else "decreased"
                    explanations.append(f"• {self.feature_explanations.get(feature, feature)}: {value} {direction} risk by {abs(impact):.2%}")
        except:
            shap_image = ""
            explanations = ["SHAP analysis requires tree-based model. Using basic prediction."]
        
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
            'shap_waterfall': shap_image,
            'explanations': explanations if explanations else ["No detailed explanations available"]
        }

# Global instances
heart_explainer = None
diabetes_explainer = None

def get_explainer(disease_type):
    global heart_explainer, diabetes_explainer
    
    if disease_type == 'heart':
        if heart_explainer is None:
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            heart_explainer = MedicalExplainer(
                'modules/module2_ml/model_heart.pkl',
                'modules/module1_data/scaler_heart.pkl',
                feature_names,
                'Heart Disease'
            )
        return heart_explainer
    
    else:  # diabetes
        if diabetes_explainer is None:
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            diabetes_explainer = MedicalExplainer(
                'modules/module2_ml/model_diabetes.pkl',
                'modules/module1_data/scaler_diabetes.pkl',
                feature_names,
                'Diabetes'
            )
        return diabetes_explainer

def predict_and_explain(patient_dict, disease_type):
    explainer = get_explainer(disease_type)
    return explainer.explain_prediction(patient_dict)