import streamlit as st
import pandas as pd
import joblib

# 1] Page Config
st.set_page_config(page_title="Titanic Survival AI", page_icon="🚢", layout="wide")

# 2] Load Model Safely
try:
    model = joblib.load('models/titanic_model.pkl')
except:
    st.error("⚠️ Titanic Model not found. Ensure 'titanic_model.pkl' is in the models folder.")
    st.stop()

st.title("🚢 Titanic Survival Prediction System")
st.write("Predicting survival probability based on passenger demographics and class.")
st.divider()

# 3] Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Identity")
    pclass = st.selectbox("Passenger Class (1st = Luxury, 3rd = Economy)", [1, 2, 3])
    sex = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 0, 80, 25)

with col2:
    st.subheader("🎟️ Journey Details")
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare (£)", min_value=0.0, value=32.0)

# 4] Industry-Grade Prediction Logic
if st.button("Predict Survival Chance", use_container_width=True):
    # A. Data Pre-processing (Matching Training logic)
    sex_val = 1 if sex == "Male" else 0
    
    # B. Features DataFrame (Ensure order matches your Titanic CSV training)
    features = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex_val,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare
    }])

    try:
        prediction = model.predict(features)
        
        st.divider()
        if prediction[0] == 1:
            st.balloons()
            st.success("✅ **Result: SURVIVED**")
            st.info("The model predicts this passenger would have likely survived the disaster.")
        else:
            st.error("❌ **Result: DID NOT SURVIVE**")
            st.warning("The model predicts high risk for this passenger profile based on historical data.")
            
    except Exception as e:
        st.error(f"Error: {e}")