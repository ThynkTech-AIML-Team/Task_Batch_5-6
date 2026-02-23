import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

familysize = sibsp + parch + 1
isalone = 0 if familysize > 1 else 1
sex = 1 if sex == "female" else 0

input_data = np.array([[pclass, sex, age, fare, sibsp, parch, familysize, isalone]])

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger is likely to SURVIVE ✅")
    else:
        st.error("Passenger is NOT likely to survive ❌")