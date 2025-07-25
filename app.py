import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Page Config
st.set_page_config(page_title="Crop Predictor 🌾", page_icon="🌱", layout="centered")

# --- Sidebar Input ---
st.sidebar.title("🌿 Enter Soil & Weather Conditions")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 70)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 60)
K = st.sidebar.slider("Potassium (K)", 5, 205, 60)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 50.0)
ph = st.sidebar.slider("Soil pH", 3.0, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# --- Main UI ---
st.title("🌾 Smart Crop Recommendation App")
st.markdown("Predict the most suitable crop to grow based on your local environment.")

st.markdown("---")

if st.button("🚀 Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop_name = label_encoder.inverse_transform(prediction)[0].capitalize()
    
    emoji_dict = {
        "rice": "🍚", "maize": "🌽", "chickpea": "🧆", "kidneybeans": "🥫",
        "pigeonpeas": "🥟", "mothbeans": "🫘", "mungbean": "🌱", "blackgram": "⚫",
        "lentil": "🥣", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
        "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
        "orange": "🍊", "papaya": "🍍", "coconut": "🥥", "cotton": "🧵",
        "jute": "🧶", "coffee": "☕"
    }

    emoji = emoji_dict.get(crop_name.lower(), "🌾")
    st.success(f"✅ Recommended Crop: **{crop_name}** {emoji}")

    st.balloons()
else:
    st.info("⬅️ Use the sliders in the sidebar to input data, then click **Predict Crop**!")

# Footer
st.markdown("---")
st.markdown("<center><small>Built with ❤️ using Streamlit</small></center>", unsafe_allow_html=True)
