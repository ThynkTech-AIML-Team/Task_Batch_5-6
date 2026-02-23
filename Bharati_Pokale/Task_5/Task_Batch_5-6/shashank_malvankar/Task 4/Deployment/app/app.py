import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("../model/house_price_model.pkl", "rb"))

st.title("House Price Prediction App")

st.write("Enter house details to predict price")

bedrooms = st.number_input("Bedrooms", 0)
bathrooms = st.number_input("Bathrooms", 0.0)
sqft_living = st.number_input("Living Area", 0)
sqft_lot = st.number_input("Lot Area", 0)
floors = st.number_input("Floors", 0.0)
waterfront = st.number_input("Waterfront (0 or 1)", 0, 1)
view = st.number_input("View", 0)
condition = st.number_input("Condition", 0)
grade = st.number_input("Grade", 0)
sqft_above = st.number_input("Sqft Above", 0)
sqft_basement = st.number_input("Sqft Basement", 0)
yr_built = st.number_input("Year Built", 1900)
yr_renovated = st.number_input("Year Renovated", 0)

house_age = 2025 - yr_built
total_size = sqft_living + sqft_lot
was_renovated = 0 if yr_renovated == 0 else 1

if st.button("Predict Price"):

    features = np.array([[
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    grade,
    sqft_above,
    sqft_basement,
    yr_built,
    house_age,
    total_size,
    was_renovated
]])

    prediction = model.predict(features)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")