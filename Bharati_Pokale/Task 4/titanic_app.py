import streamlit as st
import joblib
import pandas as pd

# App Title
st.title(" Titanic Survival Predictor")

# Load trained model and columns
model = joblib.load("titanic_model.pkl")
columns = joblib.load("titanic_columns.pkl")

st.write("Enter passenger details below:")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
family_size = st.slider("Family Size", 1, 10, 1)

# Convert gender to numeric
sex_male = 1 if sex == "male" else 0

# Create DataFrame in correct column order
input_data = pd.DataFrame(
    [[pclass, age, fare, family_size, sex_male]],
    columns=columns
)

# Prediction Button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success(" Passenger Survived")
    else:
        st.error(" Passenger Did Not Survive")