import os

import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_building", "best_engine_maintenance_model_v1.joblib"
)


@st.cache_resource
def load_model():
    """Load the trained model artifact from the repository and cache it."""
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Engine Failure Prediction App")
st.write("""
This application predicts whether an engine is likely to fail based on its operational
sensor readings. Enter the values below to get a prediction from the trained production model.
""")

engine_rpm    = st.number_input("Engine RPM", 0, 3000, 750, 10)
lub_oil_press = st.number_input("Lub Oil Pressure", 0.0, 10.0, 3.3, 0.1)
fuel_press    = st.number_input("Fuel Pressure", 0.0, 25.0, 6.7, 0.1)
coolant_press = st.number_input("Coolant Pressure", 0.0, 10.0, 2.3, 0.1)
lub_oil_temp  = st.number_input("Lub Oil Temperature", 60.0, 100.0, 77.6, 0.1)
coolant_temp  = st.number_input("Coolant Temperature", 50.0, 200.0, 78.4, 0.1)

input_data = pd.DataFrame([{
    "Engine_RPM": engine_rpm,
    "Lub_Oil_Pressure": lub_oil_press,
    "Fuel_Pressure": fuel_press,
    "Coolant_Pressure": coolant_press,
    "Lub_Oil_Temperature": lub_oil_temp,
    "Coolant_Temperature": coolant_temp,
}])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]
    result = "Faulty" if prediction == 1 else "Normal"
    message = f"The model predicts: **{result}** ({probability:.1%} confidence)"

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(message)
    else:
        st.success(message)
