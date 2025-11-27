import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load model
def load_model():
    model_path = os.path.join(os.getcwd(), "diabetes_random_forest_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title(" Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)
hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

# Aligning input to selected_features.txt
# Default missing values = 0
input_data = pd.DataFrame([{
    'age': age,
    'physical_activity_minutes_per_week': 30 if activity == "Low" else (150 if activity == "Moderate" else 300),
    'family_history_diabetes': 0,
    'glucose_fasting': glucose,
    'glucose_postprandial': glucose,
    'hba1c': hba1c,
    'diabetes_risk_score': 0,
    'diabetes_stage_No Diabetes': 1,
    'diabetes_stage_Pre-Diabetes': 0,
    'diabetes_stage_Type 2': 0
}])

# Predict
if st.button("Predict Diabetes"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"⚠ High Diabetes Risk\nProbability: {prob:.2f}%")
    else:
        st.success(f"✔ No Diabetes Detected\nProbability: {prob:.2f}%")
