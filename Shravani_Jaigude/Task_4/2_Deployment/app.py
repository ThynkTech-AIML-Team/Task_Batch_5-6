import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Predictor")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding
sex_male = 1 if sex == "male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# IMPORTANT: Columns must match training order EXACTLY
input_data = pd.DataFrame([[
    pclass,
    age,
    sibsp,
    parch,
    fare,
    sex_male,
    embarked_q,
    embarked_s
]], columns=[
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S"
])

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")