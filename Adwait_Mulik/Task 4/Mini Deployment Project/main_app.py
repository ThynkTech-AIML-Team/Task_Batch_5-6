import streamlit as st
import pandas as pd
import joblib

# 1] Global Page Config
st.set_page_config(page_title="AI Project Hub", page_icon="🚀", layout="wide")

# 2] Sidebar Navigation
st.sidebar.title("🤖 AI Project Hub")
page = st.sidebar.radio("Select a Project", ["Loan Approval", "House Valuation", "Titanic Survival"])

# --- PAGE 1: LOAN APPROVAL ---
if page == "Loan Approval":
    st.title("🏦 Loan Approval System")
    try:
        model = joblib.load('models/loan_model.pkl')
        
        col1, col2 = st.columns(2)
        with col1:
            no_of_dependents = st.number_input("Dependents", 0, 10, 1)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income (₹)", 0, 10000000, 500000)
        with col2:
            loan_amount = st.number_input("Loan Amount (₹)", 0, 10000000, 200000)
            loan_term = st.number_input("Term (Years)", 1, 20, 5)
            cibil_score = st.slider("CIBIL Score", 300, 900, 750)
            assets = st.number_input("Assets Value (₹)", 0, 10000000, 1000000)

        if st.button("Predict Loan Status", use_container_width=True):
            edu_val = 1 if education == "Graduate" else 0
            emp_val = 1 if self_employed == "Yes" else 0
            features = pd.DataFrame([[no_of_dependents, edu_val, emp_val, income_annum, loan_amount, loan_term, cibil_score, assets]], 
                                    columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value'])
            prediction = model.predict(features)
            if prediction[0] == 0:
                st.balloons(); st.success("✅ Loan Approved!")
            else:
                st.error("❌ Loan Rejected")
    except:
        st.warning("⚠️ Model not found in models/loan_model.pkl")

# --- PAGE 2: HOUSE VALUATION ---
elif page == "House Valuation":
    st.title("🏡 Bengaluru House Predictor")
    try:
        model = joblib.load('models/house_model.pkl')
        
        total_sqft = st.number_input("Total Sqft", 500, 10000, 1200)
        bhk = st.slider("BHK", 1, 5, 2)
        bath = st.slider("Bathrooms", 1, 5, 2)

        if st.button("Calculate Market Value", use_container_width=True):
            features = pd.DataFrame([[total_sqft, bath, bhk]], columns=['total_sqft', 'bath', 'bhk'])
            prediction = model.predict(features)
            st.metric("Estimated Price", f"₹ {prediction[0]:.2f} Lakhs")
    except:
        st.warning("⚠️ Model not found in models/house_model.pkl")

# --- PAGE 3: TITANIC SURVIVAL ---
elif page == "Titanic Survival":
    st.title("🚢 Titanic Survival Predictor")
    try:
        model = joblib.load('models/titanic_model.pkl')
        
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Class", [1, 2, 3])
            sex = st.radio("Sex", ["Male", "Female"])
        with col2:
            age = st.slider("Age", 0, 80, 25)
            fare = st.number_input("Fare", 0, 500, 32)

        if st.button("Predict Survival", use_container_width=True):
            sex_val = 1 if sex == "Male" else 0
            # Note: Ensure these columns match your training (e.g., Pclass, Sex, Age, SibSp, Parch, Fare)
            features = pd.DataFrame([[pclass, sex_val, age, 0, 0, fare]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.balloons(); st.success("✅ Survived!")
            else:
                st.error("❌ Did Not Survive")
    except:
        st.warning("⚠️ Model not found in models/titanic_model.pkl")