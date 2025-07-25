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
    humidity = st.number_input("Humidity (%)", 10.0, 100.0, value=70.0)
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, value=100.0)

season = st.selectbox("🌦 Current Season", ["Spring", "Summer", "Monsoon", "Autumn", "Winter"])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Best Crops"):

    # Make prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop = label_encoder.inverse_transform(prediction)[0]

    # Show top recommendation
    st.success(f"🌱 **Top Recommended Crop: {crop.capitalize()}**")

    # Season-based suggestions
    seasonal_crop_map = {
        "Spring": ["peas", "cabbage", "spinach", "carrot", "onion"],
        "Summer": ["maize", "millet", "sorghum", "soybean", "cowpea"],
        "Monsoon": ["rice", "cotton", "groundnut", "bajra", "jute"],
        "Autumn": ["wheat", "mustard", "barley", "gram", "sugarcane"],
        "Winter": ["wheat", "gram", "lentil", "mustard", "barley"]
    }

    st.markdown("🌾 **Other Suggested Crops for this Season:**")
    for s_crop in seasonal_crop_map.get(season, []):
        st.write(f"- {s_crop.capitalize()}")

    st.markdown("---")
    st.info("🚧 More features coming soon: weather auto-fill, image-based crop classifier, and downloadable reports!")

# Footer
st.markdown("---")
st.caption("🔬 Built with ❤️ using Streamlit, Scikit-learn & XGBoost")
