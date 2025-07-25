import streamlit as st
import joblib
import numpy as np
from datetime import datetime
from PIL import Image
import base64

# Load the model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Set page config
st.set_page_config(
    page_title="Crop Predictor",
    page_icon="🌾",
    layout="centered"
)

# Hero image
st.image(
    "https://cdn.pixabay.com/photo/2016/11/21/12/53/rice-1845906_1280.jpg",
    use_container_width=True
)

# App Title
st.title("🌾 Smart Crop Recommendation App")

st.markdown("""
Welcome to your smart farming assistant!  
This tool helps you choose the **best crop to grow** based on:
- Soil nutrients (N, P, K)
- Temperature
- Humidity
- pH level
- Rainfall
- 🌦 Season (for multiple suggestions)
""")

st.markdown("---")

# User inputs
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 140, value=60)
    P = st.number_input("Phosphorus (P)", 5, 145, value=60)
    K = st.number_input("Potassium (K)", 5, 205, value=60)
    ph = st.number_input("Soil pH", 3.5, 9.5, value=6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", 10.0, 50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", 10.0, 100.0, value_
