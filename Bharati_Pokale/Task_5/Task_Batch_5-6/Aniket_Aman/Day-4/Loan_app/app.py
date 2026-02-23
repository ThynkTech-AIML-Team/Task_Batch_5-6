import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Nexus Bank | Loan Assessment", page_icon="🏦", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'models/real_loan_pipeline.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

pipeline = load_model()

# --- HEADER SECTION ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) 
with col2:
    st.title("Nexus Bank Automated Loan Officer")
    st.markdown("Powered by Real-World Machine Learning Models")

st.divider()

if pipeline is None:
    st.error("⚠️ Model missing. Please run `python src/train_pipeline.py` first to download the real dataset and train the model.")
    st.stop()

# --- DASHBOARD LAYOUT ---
st.header("Applicant Profiling Form")

with st.form("loan_application_form"):
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        married = st.selectbox("Marital Status", options=["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        
    with col_b:
        st.subheader("Financials")
        self_employed = st.selectbox("Self Employed?", options=["No", "Yes"])
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0.0, value=0.0, step=500.0)
        credit_history = st.radio("Has valid Credit History?", options=["Yes", "No"])
        credit_val = 1.0 if credit_history == "Yes" else 0.0

    with col_c:
        st.subheader("Loan Specifications")
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1.0, value=150.0, step=10.0)
        loan_amount_term = st.selectbox("Loan Term (Months)", options=[12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0], index=8)
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Run Automated Underwriting", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if submitted:
    # 1. Format input data precisely as the model expects
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_val],
        'Property_Area': [property_area]
    })
    
    with st.spinner("Analyzing real-world historical data..."):
        # 2. Get prediction
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0]
        
        st.divider()
        st.header("Underwriting Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.success("### Decision: APPROVED ✅")
                st.markdown("Based on historical data, this profile is highly likely to fulfill loan obligations.")
                st.balloons()
            else:
                st.error("### Decision: REJECTED ❌")
                st.markdown("Based on historical data, this profile carries high default risk.")
                
        with res_col2:
            st.metric(label="Model Confidence", value=f"{max(probability):.1%}")
            st.progress(max(probability))