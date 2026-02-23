import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("titanic_model.pkl", "rb"))
le_sex = pickle.load(open("le_sex.pkl", "rb"))
le_embarked = pickle.load(open("le_embarked.pkl", "rb"))

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
family_size = st.slider("Family Size", 1, 10, 1)

sex = le_sex.transform([sex])[0]
embarked = le_embarked.transform([embarked])[0]

data = np.array([[pclass, sex, age, fare, embarked, family_size]])

if st.button("Predict"):
    result = model.predict(data)[0]
    if result == 1:
        st.success("Survived")
    else:
        st.error("Did Not Survive")