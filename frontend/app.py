import streamlit as st
import requests
import json
import base64
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

# API endpoint (change to your actual URL when deployed)
API_URL = "http://localhost:8000"

# Medical disclaimer
st.warning("""
⚠️ **MEDICAL DISCLAIMER**: This is an AI assistant for educational purposes only. 
Not for actual medical diagnosis. Always consult qualified healthcare professionals.
""")

st.title("🏥 AI Medical Diagnosis Assistant")
st.markdown("---")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Create tabs
tab1, tab2 = st.tabs(["📊 Disease Risk Prediction", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        disease = st.radio("Select Condition", ["Heart Disease", "Diabetes"])
        
        st.subheader("Enter Patient Information")
        
        if disease == "Heart Disease":
            age = st.slider("Age", 20, 100, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            sex_val = 1 if sex == "Male" else 0
            cp = st.select_slider("Chest Pain Type", options=[0, 1, 2, 3], value=1)
            trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
            chol = st.slider("Cholesterol", 100, 600, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0
            restecg = st.select_slider("Resting ECG", options=[0, 1, 2], value=1)
            thalach = st.slider("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
            slope = st.select_slider("ST Slope", options=[0, 1, 2], value=1)
            ca = st.slider("Major Vessels Count", 0, 3, 0)
            thal = st.select_slider("Thalassemia", options=[0, 1, 2, 3], value=1)
            
            patient_data = {
                "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs_val, "restecg": restecg, "thalach": thalach,
                "exang": exang_val, "oldpeak": oldpeak, "slope": slope,
                "ca": ca, "thal": thal
            }
            endpoint = f"{API_URL}/predict/heart"
            
        else:  # Diabetes
            pregnancies = st.number_input("Pregnancies", 0, 20, 3)
            glucose = st.slider("Glucose Level", 50, 200, 120)
            blood_pressure = st.slider("Blood Pressure", 40, 140, 70)
            skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
            insulin = st.slider("Insulin Level", 0, 900, 80)
            bmi = st.slider("BMI", 10.0, 60.0, 25.0)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.slider("Age", 20, 100, 35)
            
            patient_data = {
                "Pregnancies": pregnancies, "Glucose": glucose,
                "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }
            endpoint = f"{API_URL}/predict/diabetes"
        
        if st.button("🔍 Analyze Risk", type="primary"):
            with st.spinner("Analyzing patient data with AI..."):
                try:
                    response = requests.post(endpoint, json=patient_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Save to history
                        st.session_state.history.insert(0, {
                            "disease": disease,
                            "data": patient_data,
                            "result": result
                        })
                        st.session_state.history = st.session_state.history[:5]
                        
                        with col2:
                            # Display result
                            color = result['risk_color']
                            st.markdown(f"""
                            <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color}">
                                <h3 style="color: {color}; text-align: center;">{result['risk_level']}</h3>
                                <hr>
                                <p><strong>Probability:</strong> {result['probability']:.1%}</p>
                                <p><strong>Prediction:</strong> {'Positive' if result['prediction'] == 1 else 'Negative'}</p>
                                <p><strong>{result['risk_advice']}</strong></p>
                            </div>
                            <br>
                            """, unsafe_allow_html=True)
                            
                            # Explanations
                            st.subheader("📝 Key Risk Factors")
                            for exp in result['explanations'][:3]:
                                st.info(exp)
                            
                            # SHAP visualization
                            if result['shap_waterfall']:
                                st.subheader("🔬 AI Decision Explanation")
                                shap_img = base64.b64decode(result['shap_waterfall'])
                                st.image(Image.open(io.BytesIO(shap_img)), use_column_width=True)
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Cannot connect to API. Make sure FastAPI is running on {API_URL}")
                    st.info("Start the API with: uvicorn api.main:app --reload")
    
    # Sidebar with history
    with st.sidebar:
        st.subheader("📜 Recent Predictions")
        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history):
                with st.expander(f"{entry['disease']} - {entry['result']['risk_level']}"):
                    st.write(f"**Risk:** {entry['result']['probability']:.1%}")
                    st.write(f"**Top factor:** {entry['result']['explanations'][0][:50]}...")
        else:
            st.info("No predictions yet. Try the analyzer!")

with tab2:
    st.markdown("""
    ### 🧠 About This AI Assistant
    
    This application uses machine learning to assess disease risk based on patient data.
    
    **Features:**
    - Heart Disease Risk Prediction (AUC: 94.6%)
    - Diabetes Risk Prediction
    - SHAP explanations for each prediction
    - Plain English risk factors
    
    **Technologies:**
    - FastAPI backend
    - Streamlit frontend
    - XGBoost / Random Forest models
    - SHAP for explainability
    
    **Note:** This is a demonstration project. Always consult medical professionals.
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using AI for medical education")

st.markdown("---")
st.caption("AI Medical Diagnosis Assistant | Educational Purpose Only")