import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load trained model
model = joblib.load("digit_model.pkl")

st.title("🧠 Digit Recognition Web App")
st.write("Upload a handwritten digit image (0–9)")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    
    # Open image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))

    # Convert to numpy
    img_array = np.array(image)

    # Invert colors (MNIST format is white digit on black background)
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Flatten
    img_array = img_array.reshape(1, -1)

    # Predict
    prediction = model.predict(img_array)

    st.success(f"Predicted Digit: {prediction[0]}")