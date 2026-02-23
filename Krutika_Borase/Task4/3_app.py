import os
import joblib
import numpy as np
import pandas as pd

import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Sidebar for project info and branding
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)
    st.markdown("## Titanic Survival Predictor")
    st.markdown("Predict passenger survival on the Titanic using a trained machine learning model.")
    st.markdown("---")
    st.markdown("**Project by:** Your Name")
    st.markdown("**Model:** Logistic Regression / Random Forest / Voting Ensemble")
    st.markdown("---")
    st.markdown("[View Titanic Dataset](https://www.kaggle.com/c/titanic/data)")

st.markdown("""
<style>
.main .block-container { padding-top: 2rem; }
.stMetric { font-size: 1.2rem; }
.stSuccess { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# Titanic Survival Predictor")
st.markdown("""
Enter passenger details below to predict survival probability. The model uses key features and engineered variables for accurate prediction.
""")

MODEL_PATH = "best_titanic_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"
FEATURES_PATH = "features.pkl"

missing = [p for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, FEATURES_PATH] if not os.path.exists(p)]
if missing:
    st.error(f"Missing files: {missing}. Run your training notebook first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
features = joblib.load(FEATURES_PATH)

le_sex = encoders["sex"]
le_embarked = encoders["embarked"]
le_title = encoders["title"]

def age_group(age: float) -> int:
    if age <= 12: return 0
    if age <= 18: return 1
    if age <= 35: return 2
    if age <= 60: return 3
    return 4

# Approximate training quartiles for Titanic fare
FARE_BINS = [0.0, 7.91, 14.454, 31.0, 600.0]
def fare_bin(fare: float) -> int:
    if fare <= FARE_BINS[1]: return 0
    if fare <= FARE_BINS[2]: return 1
    if fare <= FARE_BINS[3]: return 2
    return 3

def safe_encode(le, value):
    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    return int(le.transform([classes[0]])[0])


with st.form("predict_form"):
    st.markdown("### Passenger Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", list(le_sex.classes_))
        title = st.selectbox("Title", list(le_title.classes_))
    with col2:
        age = st.slider("Age", 0, 80, 29)
        fare = st.number_input("Fare ($)", min_value=0.0, value=32.0, step=0.5)
        embarked = st.selectbox("Embarked", list(le_embarked.classes_))
    with col3:
        sibsp = st.number_input("Siblings/Spouses", min_value=0, value=0, step=1)
        parch = st.number_input("Parents/Children", min_value=0, value=0, step=1)

    st.markdown("---")
    submit = st.form_submit_button("Predict Survival", use_container_width=True)


if submit:
    family_size = int(sibsp) + int(parch) + 1
    is_alone = 1 if family_size == 1 else 0

    row = {
        "Pclass": int(pclass),
        "Sex": safe_encode(le_sex, sex),
        "Age": float(age),
        "Fare": float(fare),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Embarked": safe_encode(le_embarked, embarked),
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": safe_encode(le_title, title),
        "AgeGroup": age_group(float(age)),
        "FareBin": fare_bin(float(fare)),
        "Age*Class": float(age) * int(pclass),
        "Fare_Per_Person": float(fare) / (family_size + 1),
    }

    X_input = pd.DataFrame([row])[features]
    needs_scaled = model.__class__.__name__ in ["LogisticRegression", "SVC", "LinearSVC"]
    X_pred = scaler.transform(X_input) if needs_scaled else X_input

    pred = int(model.predict(X_pred)[0])

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_pred)[0][1])
            st.metric("Survival Probability", f"{proba:.2%}")
    with colB:
        st.success("Prediction: <b>Survived</b>" if pred == 1 else "Prediction: <b>Did Not Survive</b>", icon="✅" if pred == 1 else "❌")

    