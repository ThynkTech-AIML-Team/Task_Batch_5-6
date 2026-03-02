import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter house details below:")

# Numeric Inputs
area = st.number_input("Area (sq ft)", min_value=0.0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
floors = st.number_input("Floors", min_value=0)
year_built = st.number_input("Year Built", min_value=1800, max_value=2026)

# Categorical Inputs
location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
condition = st.selectbox("Condition", ["Fair", "Good", "Poor"])
garage = st.selectbox("Garage", ["Yes", "No"])

if st.button("Predict Price"):

    # Create base dictionary with all features set to 0
    input_data = {
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'YearBuilt': year_built,
        'Location_Rural': 0,
        'Location_Suburban': 0,
        'Location_Urban': 0,
        'Condition_Fair': 0,
        'Condition_Good': 0,
        'Condition_Poor': 0,
        'Garage_Yes': 0
    }

    # Set correct one-hot values
    input_data[f"Location_{location}"] = 1
    input_data[f"Condition_{condition}"] = 1

    if garage == "Yes":
        input_data["Garage_Yes"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)

    st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")