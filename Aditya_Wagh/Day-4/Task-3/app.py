# app.py

import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
family_size = st.slider("Family Size", 1, 10, 1)

# Convert sex
sex = 0 if sex == "male" else 1

# Prediction
if st.button("Predict Survival"):

    input_data = np.array([[pclass, sex, age, fare, family_size]])

    prediction = model.predict(input_data)

    if prediction == 1:
        st.success("Passenger is likely to SURVIVE")
    else:
        st.error("Passenger is NOT likely to survive")
