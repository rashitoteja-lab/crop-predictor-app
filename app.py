
import streamlit as st
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

st.title("🌾 Crop Recommendation App")
st.markdown("Enter environmental and soil conditions:")

N = st.number_input("Nitrogen", 0, 140)
P = st.number_input("Phosphorus", 5, 145)
K = st.number_input("Potassium", 5, 205)
temperature = st.number_input("Temperature (°C)", 10.0, 50.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0)
ph = st.number_input("pH", 3.0, 10.0)
rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0)

if st.button("Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop_name = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Recommended crop: **{crop_name.capitalize()}**")
