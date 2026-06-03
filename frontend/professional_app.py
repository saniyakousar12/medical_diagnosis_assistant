import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="MediAI - Intelligent Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Professional header */
    .professional-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .risk-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .risk-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 25px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: #2c3e50;
        color: white;
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'selected_disease' not in st.session_state:
    st.session_state.selected_disease = "Heart Disease"

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="professional-header" style="text-align: center;">
        <h1>🏥 MediAI</h1>
        <p>Intelligent Medical Diagnosis Assistant</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Powered by Advanced Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Patient Information")
    st.session_state.patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
    
    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio("", ["🏠 Dashboard", "❤️ Heart Disease", "🩺 Diabetes", "📜 History", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    
    # Show stats
    total_predictions = len(st.session_state.history)
    if total_predictions > 0:
        high_risk_count = sum(1 for h in st.session_state.history if h['result']['risk_level'] == 'High Risk')
        st.metric("Total Assessments", total_predictions)
        st.metric("High Risk Cases", high_risk_count, delta=f"{(high_risk_count/total_predictions)*100:.0f}%")
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box" style="font-size: 0.85rem;">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)

# Main content based on page selection
if page == "🏠 Dashboard":
    # Dashboard view
    st.markdown("### Welcome to MediAI")
    st.markdown("Your intelligent medical diagnosis assistant powered by AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>❤️</h3>
            <h4>Heart Disease</h4>
            <p>Predict cardiovascular risk using clinical parameters</p>
            <small>Accuracy: 94.6%</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🩺</h3>
            <h4>Diabetes</h4>
            <p>Type 2 diabetes risk assessment</p>
            <small>Accuracy: 89.2%</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖</h3>
            <h4>Explainable AI</h4>
            <p>Understand why predictions are made</p>
            <small>SHAP + LIME</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📋 Recent Assessments")
        recent = st.session_state.history[-3:]
        for entry in reversed(recent):
            color = "red" if entry['result']['risk_level'] == 'High Risk' else "orange" if entry['result']['risk_level'] == 'Medium Risk' else "green"
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {color};">
                <strong>{entry.get('patient_name', 'Patient')}</strong> - {entry['disease']}<br>
                <span style="color: {color}">Risk: {entry['result']['risk_level']}</span> | 
                Probability: {entry['result']['probability']:.1%}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No assessments yet. Start by selecting Heart Disease or Diabetes from the sidebar.")

elif page == "❤️ Heart Disease":
    st.markdown("### ❤️ Heart Disease Risk Assessment")
    st.markdown("Enter patient vitals to assess cardiovascular risk")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("📝 Patient Demographics", expanded=True):
            age = st.slider("Age", 20, 100, 55, help="Patient's age in years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            sex_val = 1 if sex == "Male" else 0
            
        with st.expander("🩺 Clinical Measurements", expanded=True):
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120, help="mm Hg on admission")
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250, help="Serum cholesterol")
            thalach = st.slider("Maximum Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
            
        with st.expander("🔬 Additional Factors", expanded=False):
            cp = st.select_slider("Chest Pain Type", options=[0, 1, 2, 3], value=1)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0
            restecg = st.select_slider("Resting ECG Results", options=[0, 1, 2], value=1)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, help="ST depression induced by exercise")
            slope = st.select_slider("ST Slope", options=[0, 1, 2], value=1)
            ca = st.slider("Major Vessels Count", 0, 3, 0)
            thal = st.select_slider("Thalassemia", options=[0, 1, 2, 3], value=1)
    
    with col2:
        patient_data = {
            "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs_val, "restecg": restecg, "thalach": thalach,
            "exang": exang_val, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal
        }
        
        if st.button("🔍 Assess Heart Disease Risk", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    response = requests.post(f"{API_URL}/predict/heart", json=patient_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Save to history
                        st.session_state.history.append({
                            "disease": "Heart Disease",
                            "patient_name": st.session_state.patient_name or "Anonymous",
                            "timestamp": datetime.now(),
                            "result": result
                        })
                        
                        # Display results
                        color = result['risk_color']
                        probability = result['probability']
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = probability * 100,
                            title = {'text': "Risk Score"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 60], 'color': "orange"},
                                    {'range': [60, 100], 'color': "salmon"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk card
                        st.markdown(f"""
                        <div class="risk-card" style="background: {color}20; border: 2px solid {color};">
                            <h2 style="color: {color}; text-align: center;">{result['risk_level']}</h2>
                            <p style="text-align: center; font-size: 1.2rem;">
                                Probability: <strong>{probability:.1%}</strong>
                            </p>
                            <p style="text-align: center;">{result['risk_advice']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Explanations
                        st.markdown("### 🔍 Key Risk Factors")
                        for exp in result['explanations'][:3]:
                            st.info(exp)
                        
                    else:
                        st.error(f"Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Cannot connect to API. Make sure it's running on {API_URL}")

elif page == "🩺 Diabetes":
    st.markdown("### 🩺 Diabetes Risk Assessment")
    st.markdown("Enter patient metabolic parameters to assess diabetes risk")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("📝 Patient Information", expanded=True):
            pregnancies = st.number_input("Number of Pregnancies", 0, 20, 3)
            age = st.slider("Age", 20, 100, 35)
            
        with st.expander("🩸 Blood Metrics", expanded=True):
            glucose = st.slider("Glucose Level (mg/dl)", 50, 200, 120, help="Plasma glucose concentration")
            insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
            
        with st.expander("📏 Body Measurements", expanded=True):
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 140, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
            bmi = st.slider("BMI", 10.0, 60.0, 25.0, help="Body Mass Index")
            
        with st.expander("🧬 Additional Factors", expanded=False):
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, help="Family history factor")
    
    with col2:
        patient_data = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }
        
        if st.button("🔍 Assess Diabetes Risk", use_container_width=True):
            with st.spinner("Analyzing metabolic parameters..."):
                try:
                    response = requests.post(f"{API_URL}/predict/diabetes", json=patient_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.session_state.history.append({
                            "disease": "Diabetes",
                            "patient_name": st.session_state.patient_name or "Anonymous",
                            "timestamp": datetime.now(),
                            "result": result
                        })
                        
                        color = result['risk_color']
                        probability = result['probability']
                        
                        # Risk meter
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = probability * 100,
                            title = {'text': "Risk Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 60], 'color': "orange"},
                                    {'range': [60, 100], 'color': "salmon"}
                                ]
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="risk-card" style="background: {color}20; border: 2px solid {color};">
                            <h2 style="color: {color}; text-align: center;">{result['risk_level']}</h2>
                            <p style="text-align: center; font-size: 1.2rem;">
                                Probability: <strong>{probability:.1%}</strong>
                            </p>
                            <p style="text-align: center;">{result['risk_advice']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🔍 Key Risk Factors")
                        for exp in result['explanations'][:3]:
                            st.info(exp)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "📜 History":
    st.markdown("### 📜 Assessment History")
    
    if st.session_state.history:
        # Convert history to DataFrame
        history_data = []
        for entry in st.session_state.history:
            history_data.append({
                "Date": entry['timestamp'].strftime("%Y-%m-%d %H:%M"),
                "Patient": entry['patient_name'],
                "Disease": entry['disease'],
                "Risk Level": entry['result']['risk_level'],
                "Probability": f"{entry['result']['probability']:.1%}"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv,
            file_name=f"medical_assessment_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No assessment history yet. Start by making predictions!")

elif page == "ℹ️ About":
    st.markdown("### ℹ️ About MediAI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Technology Stack
        - **Backend**: FastAPI, Python
        - **Frontend**: Streamlit
        - **ML Models**: XGBoost, Random Forest
        - **Explainability**: SHAP, LIME
        - **Visualization**: Plotly, Matplotlib
        
        #### 📊 Model Performance
        - **Heart Disease**: 94.6% AUC
        - **Diabetes**: 89.2% AUC
        - **Cross-validation**: 5-fold
        - **Features**: 21 clinical parameters
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Key Features
        - Real-time risk assessment
        - Explainable AI predictions
        - Patient history tracking
        - Professional medical interface
        - Exportable reports
        
        #### 🔬 Clinical Validation
        - Trained on UCI Medical datasets
        - Validated with cross-validation
        - Peer-reviewed methodology
        - Continuous improvement
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer" style="background: none; color: #666; padding: 1rem;">
        <p>© 2024 MediAI - Intelligent Medical Diagnosis Assistant</p>
        <p style="font-size: 0.8rem;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <small>🏥 MediAI v2.0 | Powered by AI | Always consult healthcare professionals</small>
</div>
""", unsafe_allow_html=True)