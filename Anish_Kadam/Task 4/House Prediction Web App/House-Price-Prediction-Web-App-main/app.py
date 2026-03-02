import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/house_price_model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# Numeric Inputs
area = st.number_input("Area (sq ft)", min_value=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1)
floors = st.number_input("Number of Floors", min_value=1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025)

# Dropdown Inputs
location = st.selectbox(
    "Location",
    ["Rural", "Suburban", "Urban", "Downtown"]
)

condition = st.selectbox(
    "Condition",
    ["Excellent", "Good", "Fair", "Poor"]
)

garage = st.selectbox(
    "Garage",
    ["Yes", "No"]
)

if st.button("Predict Price"):
    
    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "YearBuilt": [year_built],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: $ {round(prediction, 2)}")