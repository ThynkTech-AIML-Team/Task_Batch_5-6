import streamlit as st
import numpy as np
import joblib

model = joblib.load("titanic_model.pkl")
st.title("Titanic survival predictor")
st.write("Enter passenger details below:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Number of Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

sex = 0 if sex == "Male" else 1

input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

if st.button("Predict survival"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Paasenger would survive.")
    else:
        st.error("Passenger would not survive.")
