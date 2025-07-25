import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Page config
st.set_page_config(page_title="Crop Predictor 🌾", page_icon="🌱", layout="centered")

# Hero image
st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994", use_column_width=True, caption="Smart Farming for a Sustainable Future 🌱")

# Title and intro
st.title("🌾 Crop Recommendation App")
st.markdown("Enter your soil and climate parameters to get the best crop recommendation for your conditions.")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("🧪 Soil & Climate Inputs")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 70)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 60)
K = st.sidebar.slider("Potassium (K)", 5, 205, 60)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.sidebar.slider("Soil pH", 3.0, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Visual: Nutrient levels
st.subheader("🌱 Nutrient Levels (NPK)")
npk_df = pd.DataFrame({
    'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
    'Level': [N, P, K]
})
st.bar_chart(npk_df.set_index('Nutrient'))

# Info section
with st.expander("ℹ️ What do these values mean?"):
    st.write("""
    - **Nitrogen (N):** Helps plants grow fast, increases seed and fruit production.
    - **Phosphorus (P):** Stimulates root development and flowering.
    - **Potassium (K):** Enhances drought resistance and disease tolerance.
    - **Temperature & Humidity:** Environmental conditions for crop suitability.
    - **pH:** Indicates acidity or alkalinity of soil.
    - **Rainfall:** Annual rainfall in your region (in mm).
    """)

st.markdown("---")

# Prediction logic
if st.button("🚀 Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop_name = label_encoder.invers
