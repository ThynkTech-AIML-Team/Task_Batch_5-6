import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("titanic_survival_model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare Paid", min_value=0.0, value=10.0)
family_size = st.slider("Family Size", 1, 10, 1)
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

# Encoding
sex = 0 if sex == "Male" else 1
embarked_map = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
embarked = embarked_map[embarked]
fare_per_person = fare / family_size
is_alone = 1 if family_size == 1 else 0

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, fare_per_person, family_size, is_alone, embarked]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to SURVIVE")
    else:
        st.error("❌ The passenger is NOT likely to survive")