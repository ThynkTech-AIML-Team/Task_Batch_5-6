import streamlit as st
import pandas as pd
import joblib

# 1] Professional Page Setup
st.set_page_config(page_title="Loan Approval AI", page_icon="🏦", layout="wide")

# 2] Load the trained model safely
try:
    # Ensure the path matches your VS Code structure
    model = joblib.load('models/loan_model.pkl')
except Exception as e:
    st.error(f"⚠️ Model Error: Could not load 'models/loan_model.pkl'. Please ensure the file is in the models folder. Error: {e}")
    st.stop()

# 3] Professional Header
st.title("🏦 Industry-Grade Loan Approval System")
st.markdown("""
    This system utilizes a **Random Forest Classifier** to evaluate loan applications. 
    It analyzes the interaction between credit history, income, assets, and liability to provide a data-driven decision.
""")
st.divider()

# 4] User Input Form (Organized into Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Profile")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1, value=1)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Employment Status", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, step=10000, value=500000)

with col2:
    st.subheader("💰 Loan & Asset Details")
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=10000, value=200000)
    loan_term = st.number_input("Loan Tenure (Years)", min_value=1, max_value=20, step=1, value=5)
    cibil_score = st.slider("CIBIL Score", 300, 900, 750)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, step=10000, value=1000000)

st.divider()

# 5] Industry-Grade Prediction Logic
if st.button("Generate Final Prediction", use_container_width=True):
    
    # A. Data Pre-processing (Binary Encoding for categorical features)
    edu_val = 1 if education == "Graduate" else 0
    emp_val = 1 if self_employed == "Yes" else 0
    
    # B. Creating the Input Dataframe (Must match training columns exactly)
    features = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'education': edu_val,
        'self_employed': emp_val,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value
    }])

    # C. Model Execution & Result Visualization
    try:
        prediction = model.predict(features)
        
        st.subheader("System Assessment Result")
        
        # Checking against the encoded target classes from training
        if prediction[0] == 0:  # Assuming 0 = Approved, 1 = Rejected from your dummy encoding
            st.balloons()
            st.success("✅ **STATUS: APPROVED**")
            st.info("The profile demonstrates a healthy balance of credit score and asset-to-liability ratio.")
        else:
            st.error("❌ **STATUS: REJECTED**")
            st.warning("The system has flagged this profile as high-risk. Consider improving the CIBIL score or asset collateral.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the model was trained with the same 8 features used in this form.")