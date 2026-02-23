import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("../outputs/processed_data/best_model.pkl", "rb"))

st.title("Titanic Survival Predictor")

st.write("Enter passenger details:")

# Inputs
Pclass = st.selectbox("Passenger Class", [1,2,3])

Sex = st.selectbox("Sex", ["Male","Female"])

Age = st.slider("Age", 1, 80, 25)

SibSp = st.slider("Siblings/Spouses", 0, 8, 0)

Parch = st.slider("Parents/Children", 0, 6, 0)

Fare = st.slider("Fare", 0.0, 500.0, 50.0)

Embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

FamilySize = SibSp + Parch + 1

# Convert Sex
Sex = 0 if Sex == "Male" else 1

# Convert Embarked
Embarked_Q = 0
Embarked_S = 0

if Embarked == "Queenstown":
    Embarked_Q = 1
elif Embarked == "Southampton":
    Embarked_S = 1

# Create dataframe
input_data = pd.DataFrame({

    "Pclass":[Pclass],
    "Sex":[Sex],
    "Age":[Age],
    "SibSp":[SibSp],
    "Parch":[Parch],
    "Fare":[Fare],
    "FamilySize":[FamilySize],
    "Embarked_Q":[Embarked_Q],
    "Embarked_S":[Embarked_S]

})

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger will SURVIVE")
    else:
        st.error("Passenger will NOT SURVIVE")