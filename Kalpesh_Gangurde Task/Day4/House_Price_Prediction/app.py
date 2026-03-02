import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd


# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title(" House Price Prediction")
st.write("Enter the property details below to predict the house price.")
st.divider()

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try loading with joblib first
        try:
            model = joblib.load('model.pkl')
        except:
            # Fallback to pickle
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'model.pkl' is in the same directory as this app.")
        return None

model = load_model()

if model is not None:
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Property Details")
        
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input(
                "Area (sq ft)",
                min_value=0.0,
                step=100.0,
                value=2000.0,
                help="Enter the property area in square feet"
            )
        
        with col2:
            bedrooms = st.number_input(
                "Number of Bedrooms",
                min_value=0,
                max_value=20,
                step=1,
                value=3,
                help="Number of bedrooms in the property"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            bathrooms = st.number_input(
                "Number of Bathrooms",
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                value=2.0,
                help="Number of bathrooms in the property"
            )
        
        with col4:
            location = st.selectbox(
                "Location",
                ["Downtown", "Suburbs", "Rural", "Waterfront"],
                help="Select the location of the property"
            )
        
        # Location encoding
        location_mapping = {
            "Downtown": 0,
            "Suburbs": 1,
            "Rural": 2,
            "Waterfront": 3
        }
        location_encoded = location_mapping[location]
        
        st.divider()
        
        # Submit button
        submit_button = st.form_submit_button(
            label="🔮 Predict Price",
            use_container_width=True
        )
        
        if submit_button:
            # Input validation
            if area <= 0:
                st.error("❌ Area must be greater than 0")
            elif bedrooms < 0 or bathrooms < 0:
                st.error("❌ Number of bedrooms and bathrooms cannot be negative")
            else:
                try:
                    # Prepare input features for the model
                    # Features: [Area, Bedrooms, Bathrooms, Location]
                    features = np.array([[area, bedrooms, bathrooms, location_encoded]])
                    
                    # Make prediction
                    predicted_price = model.predict(features)[0]
                    
                    # Display result
                    st.markdown(f"""
                    <div class="result-box">
                        <h3 style="margin: 0; color: #155724;">Predicted House Price</h3>
                        <h1 style="margin: 10px 0 0 0; color: #28a745;">₹{predicted_price:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display input summary
                    with st.expander("📋 Prediction Details"):
                        st.write(f"**Area:** {area:,.0f} sq ft")
                        st.write(f"**Bedrooms:** {bedrooms}")
                        st.write(f"**Bathrooms:** {bathrooms}")
                        st.write(f"**Location:** {location}")
                        st.write(f"**Predicted Price:** ₹{predicted_price:,.2f}")
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
else:
    st.info("Model could not be loaded. Please check the 'model.pkl' file.")
