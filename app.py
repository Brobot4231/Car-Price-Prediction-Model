import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Page Configurations
# -------------------------------
st.set_page_config(page_title="Car Valuation Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Selling Price Predictor")
st.write("Select the vehicle specifications below to estimate its current market value.")

# -------------------------------
# Load ML Assets
# -------------------------------
@st.cache_resource  # Keeps assets in memory for optimal loading speeds
def load_ml_assets():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:   # corrected filename
        encoder_dict = pickle.load(f)
    return model, encoder_dict

try:
    lr_model, encoders = load_ml_assets()
except FileNotFoundError:
    st.error("Error: 'car_price_model.pkl' or 'encoders.pkl' not found. Please ensure files exist in this directory.")
    st.stop()

# -------------------------------
# Extract options dynamically from encoders
# -------------------------------
brands_list = list(encoders['name'].classes_)
fuels_list = list(encoders['fuel'].classes_)
sellers_list = list(encoders['seller_type'].classes_)
owners_list = list(encoders['owner'].classes_)
transmissions_list = list(encoders['transmission'].classes_)

# -------------------------------
# Split UI Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    user_brand = st.selectbox("Car Brand / Model", brands_list)
    user_year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018, step=1)
    user_km = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
    user_fuel = st.selectbox("Fuel Type", fuels_list)
    user_seller = st.selectbox("Seller Type", sellers_list)

with col2:
    user_trans = st.selectbox("Transmission Type", transmissions_list)
    user_owner = st.selectbox("Owner History", owners_list)
    user_mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=19.5, step=0.1)
    user_engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=6000, value=1248, step=10)
    user_power = st.number_input("Max Power (BHP)", min_value=10.0, max_value=1000.0, value=85.0, step=1.0)

user_seats = st.slider("Number of Seats", min_value=2, max_value=10, value=5)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Calculate Estimated Value", type="primary"):
    # Encode categorical inputs using fitted LabelEncoders
    encoded_brand = encoders['name'].transform([user_brand])[0]
    encoded_fuel = encoders['fuel'].transform([user_fuel])[0]
    encoded_seller = encoders['seller_type'].transform([user_seller])[0]
    encoded_owner = encoders['owner'].transform([user_owner])[0]
    encoded_trans = encoders['transmission'].transform([user_trans])[0]

    # Assemble input DataFrame
    input_df = pd.DataFrame([{
        'name': encoded_brand,
        'year': user_year,
        'km_driven': user_km,
        'fuel': encoded_fuel,
        'seller_type': encoded_seller,
        'transmission': encoded_trans,
        'owner': encoded_owner,
        'mileage': user_mileage,
        'engine': user_engine,
        'max_power': user_power,
        'seats': int(user_seats)
    }])

    # Predict
    predicted_log_price = lr_model.predict(input_df)[0]
    final_price = np.expm1(predicted_log_price)

    # Display result
    st.markdown("---")
    if final_price > 0:
        st.success(f"### 💰 Estimated Market Value: **₹{final_price:,.0f}**")
    else:
        st.warning("⚠️ The entered configuration resulted in an invalid valuation.")
