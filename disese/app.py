import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model and scaler with caching ===
@st.cache_resource
def load_artifacts():
    model = joblib.load("high_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# === App UI ===
st.title("🩺 Chronic Disease Risk Prediction App")
st.markdown("Fill in the health details below to predict your risk level.")

# === User Inputs ===
age = st.number_input("Age", min_value=1, max_value=120, value=40)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
bp = st.number_input("Blood Pressure (mmHg)", min_value=70, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=70, max_value=400, value=90)

# === Create Input DataFrame ===
input_data = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "Blood_Pressure": bp,
    "Cholesterol": chol,
    "Glucose_Level": glucose
}])

# === Make Prediction ===
if st.button("Predict Risk"):
    try:
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict risk (assuming classifier with predict_proba)
        risk_score = model.predict_proba(input_scaled)[:, 1][0]
        risk_percent = risk_score * 100

        # Display Result
        st.subheader("🔍 Prediction Result")
        st.metric(label="Risk Score", value=f"{risk_percent:.1f}%")
        
        if risk_score > 0.5:
            st.markdown("### 🟥 High Risk")
        else:
            st.markdown("### 🟩 Low Risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
