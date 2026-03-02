import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page title
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("🔢 MNIST Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and let the CNN model predict it!")

# --- THE FIX: Absolute Pathing to .keras ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ IMPORTANT: We are explicitly pointing to the .keras file now, not .h5!
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cnn_digit_model.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Debug message so you can see exactly where Python is looking
st.warning(f"Looking for model at: {MODEL_PATH}")

# Notice I removed @st.cache_resource to completely clear Streamlit's memory!
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Safety check: Only load if the file actually exists
if os.path.exists(MODEL_PATH):
    try:
        model = load_my_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        model = None
else:
    st.error("❌ Cannot load model. The file does not exist. Did you run train.py?")
    model = None

# Sidebar info
st.sidebar.header("About")
st.sidebar.text("Model: CNN (Deep Learning)")
st.sidebar.text("Dataset: MNIST")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Only show the prediction UI if the file uploaded AND the model loaded successfully
if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', width=150)
    
    # Preprocessing to match training data
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (28, 28))
    img_normalized = img_resized / 255.0
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    if st.button('Predict'):
        prediction = model.predict(img_final)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.bar_chart(prediction[0])