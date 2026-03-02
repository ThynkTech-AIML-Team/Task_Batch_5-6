import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
stories = st.number_input("Number of floors", min_value=1, max_value=5)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories]])
    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")