import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

def load_model():
    model_path = os.path.join(os.getcwd(), "diabetes_random_forest_model.pkl")

    try:
        # First attempt normal pickle load
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning("Normal pickle load failed, trying alternate import…")

        # Safe fallback loader for sklearn objects
        import importlib
        import sklearn.ensemble
        import sklearn.preprocessing

        pickle.Unpickler.find_class = lambda self, module, name: \
            getattr(importlib.import_module(module), name)

        with open(model_path, "rb") as f:
            return pickle.load(f)

model = load_model()

st.title(" Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# Inputs
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
glucose = st.number_input("Glucose Level", 50, 300, 100)
hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Non-Smoker", "Smoker"])
activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

# Align to model expected features
input_data = pd.DataFrame([{
    'age': age,
    'glucose_fasting': glucose,
    'glucose_postprandial': glucose,
    'hba1c': hba1c,
    'physical_activity_minutes_per_week': 150 if activity=="Moderate" else (300 if activity=="High" else 30),
    'family_history_diabetes': 0,
    'diabetes_risk_score': 0,
    'diabetes_stage_No Diabetes': 1,
    'diabetes_stage_Pre-Diabetes': 0,
    'diabetes_stage_Type 2': 0
}])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"⚠ Diabetes Detected — Risk: {prob:.2f}%")
    else:
        st.success(f"✔ No Diabetes — Risk: {prob:.2f}%")
