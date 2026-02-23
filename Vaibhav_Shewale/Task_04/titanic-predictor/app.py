import streamlit as st
import os
st.write("Current working directory:", os.getcwd())
st.write("Files in current dir:", os.listdir('.'))
st.write("model folder exists?", os.path.isdir("model"))
if os.path.isdir("model"):
    st.write("Files in model/:", os.listdir("model"))
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🛳️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    file_path = "model/titanic_model_full_pipeline.pkl"
    
    import os
    st.write("**Debug – Current dir:**", os.getcwd())
    st.write("**Debug – Looking for:**", os.path.abspath(file_path))
    st.write("**Debug – File exists?**", os.path.exists(file_path))
    st.write("**Debug – Is file?**", os.path.isfile(file_path))
    if os.path.exists(file_path):
        st.write("**Debug – File size (bytes):**", os.path.getsize(file_path))
    else:
        st.write("**Debug – Files actually in model/**", os.listdir("model") if os.path.isdir("model") else "folder missing")
    
    try:
        model = joblib.load(file_path)
        st.success("Model loaded successfully (debug mode)")
        return model
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        st.stop()
model = load_model()

st.title("🛳️ Titanic Survival Predictor")
st.markdown("""
Would you have survived the Titanic disaster?  
Fill in your information and find out what the model predicts!
""")


st.sidebar.header("About this app")
st.sidebar.markdown("""
- Model: Random Forest Classifier
- Training data: classic Kaggle Titanic dataset (~891 passengers)
- Features used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Accuracy (test set): ~0.80–0.84 (depending on random state)
""")

st.sidebar.info("Created Feb 2026 • Streamlit + scikit-learn")


st.subheader("Your Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Ticket class (1 = best, 3 = cheapest)",
        options=[1, 2, 3],
        format_func=lambda x: f"Class {x}"
    )

    sex = st.radio("Sex", ["male", "female"], horizontal=True)

    age = st.slider("Age", 0.5, 80.0, 28.0, 0.5)

with col2:
    sibsp = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children aboard", 0, 6, 0)
    fare = st.number_input("Ticket Fare ($)", 0.0, 550.0, 32.0, step=1.0)

    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S", "Unknown"])

input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked if embarked != "Unknown" else np.nan
}])

if st.button("Will I Survive? 🧐", type="primary", use_container_width=True):

    with st.spinner("The crew is checking the records..."):
        proba = model.predict_proba(input_data)[0]
        pred_class = model.predict(input_data)[0]

        survived_prob = proba[1] * 100

    st.divider()

    if pred_class == 1:
        st.success(f"**SURVIVED** — {survived_prob:.1f}% probability")
        st.balloons()
    else:
        st.error(f"**DID NOT SURVIVE** — only {survived_prob:.1f}% chance")

    st.progress(survived_prob / 100)

    st.caption(f"Prediction made at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")

    with st.expander("See probability breakdown & feature influence"):
        colA, colB = st.columns([3,2])

        with colA:
            st.metric("Survival probability", f"{survived_prob:.1f}%")
            st.metric("Death probability", f"{proba[0]*100:.1f}%")

        with colB:
            st.markdown("**Most important factors (general model)**")
            st.markdown("• Sex (female → ↑)")
            st.markdown("• Pclass (1st → ↑↑)")
            st.markdown("• Fare (higher → ↑)")
            st.markdown("• Age (young → slight ↑)")

st.markdown("---")
st.caption("Disclaimer: This is a statistical toy model from 1912 data — not destiny!")