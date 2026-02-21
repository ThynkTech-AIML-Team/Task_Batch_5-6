import streamlit as st
import pickle
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load features
with open(FEATURES_PATH, "rb") as f:
    features_list = pickle.load(f)

# Streamlit UI
st.title("House Price Prediction App")
st.write("Enter house details to predict the price:")

# User input
bedrooms = st.number_input("Bedrooms", 0)
bathrooms = st.number_input("Bathrooms", 0.0)
sqft_living = st.number_input("Living Area", 0)
sqft_lot = st.number_input("Lot Area", 0)
floors = st.number_input("Floors", 0.0)
waterfront = st.number_input("Waterfront (0 or 1)", 0, 1)
view = st.number_input("View", 0)
condition = st.number_input("Condition", 0)
sqft_above = st.number_input("Sqft Above", 0)
sqft_basement = st.number_input("Sqft Basement", 0)
yr_built = st.number_input("Year Built", 1900)
yr_renovated = st.number_input("Year Renovated", 0)

# Prepare DataFrame
input_dict = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement,
    'yr_built': yr_built,
    'yr_renovated': yr_renovated
}

input_df = pd.DataFrame([input_dict])

# Reorder to match training
input_df = input_df.reindex(columns=features_list, fill_value=0)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")