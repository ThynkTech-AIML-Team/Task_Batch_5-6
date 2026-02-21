import streamlit as st
import pandas as pd
import joblib

# 1. Set page configuration for a wider, cleaner look
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# 2. Load the trained model
# @st.cache_resource ensures the model is only loaded once and not on every button click
@st.cache_resource 
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()

# 3. Build the UI Headers
st.title("🏠 House Price Prediction App")
st.markdown("""
Welcome to the House Price Predictor! 
Adjust the sliders below to define the features of the house, and our machine learning model will estimate its price.
""")

st.divider()

# 4. Input features using Streamlit widgets organized in columns
st.header("🏡 Property Features")

col1, col2 = st.columns(2)

with col1:
    size_sqft = st.slider("Size (in sq. ft.)", min_value=500, max_value=5000, value=1500, step=50)
    bedrooms = st.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)

with col2:
    age_years = st.slider("Age of Property (in years)", min_value=0, max_value=50, value=10, step=1)

# Format inputs into a DataFrame exactly how the model expects it
input_data = pd.DataFrame({
    'Size_sqft': [size_sqft],
    'Bedrooms': [bedrooms],
    'Age_years': [age_years]
})

st.divider()

# 5. Prediction button and logic
if st.button("Predict Price 📊", type="primary"):
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result with a success banner
    st.success(f"### Estimated House Price: ${prediction:,.2f}")
    
    # Add a fun Streamlit animation!
    st.balloons()