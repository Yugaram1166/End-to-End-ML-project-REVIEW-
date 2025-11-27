import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import pickle
import os

model_path = os.path.join(os.getcwd(), "diabetes_random_forest_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# User Inputs UI
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)
hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=350, value=180)
systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

# Load selected feature names
features_path = os.path.join(os.getcwd(), "selected_features.txt")
with open(features_path, "r") as f:
    selected_features = [line.strip() for line in f]

# Create aligned input dict
input_dict = {feat: 0 for feat in selected_features}

# Mapping available inputs
if 'age' in input_dict: input_dict['age'] = age
if 'hba1c' in input_dict: input_dict['hba1c'] = hba1c

if 'glucose_fasting' in input_dict: input_dict['glucose_fasting'] = glucose
if 'glucose_postprandial' in input_dict: input_dict['glucose_postprandial'] = glucose

if 'physical_activity_minutes_per_week' in input_dict:
    activity_map = {"Low": 30, "Moderate": 150, "High": 300}
    input_dict['physical_activity_minutes_per_week'] = activity_map[physical_activity]

# DataFrame in correct order
input_aligned = pd.DataFrame([input_dict], columns=selected_features)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_aligned)[0]
    result = "✔ Diabetes Detected" if prediction == 1 else "❌ No Diabetes"
    st.success(result)

