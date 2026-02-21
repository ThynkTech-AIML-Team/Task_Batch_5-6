import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = sns.load_dataset("titanic")

# Select important columns
df = df[['survived','pclass','sex','age','sibsp','parch','fare','embarked']]

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Define X and y
X = df.drop('survived', axis=1)
y = df['survived']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert user input into dataframe
input_data = pd.DataFrame({
    'pclass': [pclass],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'sex_male': [1 if sex == "male" else 0],
    'embarked_Q': [1 if embarked == "Q" else 0],
    'embarked_S': [1 if embarked == "S" else 0]
})

# Ensure all columns exist
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[X.columns]

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🎉 The passenger is likely to SURVIVE.")
    else:
        st.error("⚠ The passenger is likely NOT to survive.")
        