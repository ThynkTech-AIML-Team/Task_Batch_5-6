import streamlit as st
import pandas as pd
import joblib

# 1] Page Config
st.set_page_config(page_title="Bengaluru House Valuation", page_icon="🏡", layout="wide")

# 2] Load Model Safely
try:
    model = joblib.load('models/house_model.pkl')
except:
    st.error("⚠️ House Model not found in 'models/' folder.")
    st.stop()

st.title("🏡 Bengaluru Real Estate Valuation AI")
st.write("Predicting property prices based on current market trends and area metrics.")
st.divider()

# 3] Professional Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Property Location & Size")
    location = st.text_input("Enter Area (e.g., Whitefield, Indiranagar)", value="Whitefield")
    total_sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, value=1200)

with col2:
    st.subheader("🏗️ Construction Details")
    bhk = st.slider("BHK (Bedrooms)", 1, 5, 2)
    bath = st.slider("Bathrooms", 1, 5, 2)

# 4] Industry-Grade Prediction Logic
if st.button("Calculate Market Value", use_container_width=True):
    # A. Data Pre-processing
    # Ensure the features match the order in your House Price CSV!
    features = pd.DataFrame([{
        'total_sqft': total_sqft,
        'bath': bath,
        'bhk': bhk
    }])

    try:
        prediction = model.predict(features)
        price_lakhs = prediction[0]
        
        st.divider()
        st.metric(label="Estimated Property Value", value=f"₹ {price_lakhs:.2f} Lakhs")
        
        # Professional breakdown
        st.info(f"At {total_sqft} sqft, the rate is approx ₹{(price_lakhs*100000)/total_sqft:.2f} per sqft.")
    except Exception as e:
        st.error(f"Error: {e}")