import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details to predict survival.")

# User Inputs
Pclass = st.selectbox("Passenger Class", [1,2,3])
Sex = st.selectbox("Sex", ["male","female"])
Age = st.slider("Age", 0,80,25)
SibSp = st.slider("Siblings/Spouses Aboard",0,5,0)
Parch = st.slider("Parents/Children Aboard",0,5,0)
Fare = st.slider("Fare",0,500,50)
Embarked = st.selectbox("Embarked",["S","C","Q"])

# Convert inputs to model format
Sex = 0 if Sex=="male" else 1

if Embarked=="S":
    Embarked=0
elif Embarked=="C":
    Embarked=1
else:
    Embarked=2

# Prediction button
if st.button("Predict Survival"):

    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    survive_prob = probability[0][1] * 100
    not_survive_prob = probability[0][0] * 100

    if prediction[0]==1:
        st.success(f"🎉 Passenger is likely to SURVIVE")
        st.info(f"🧠 Survival Probability: {survive_prob:.2f}%")
    else:
        st.error("❌ Passenger is NOT likely to survive")
        st.info(f"🧠 Survival Probability: {survive_prob:.2f}%")