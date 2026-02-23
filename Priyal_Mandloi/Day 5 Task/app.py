import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    layout="centered"
)

st.title("🧠 MNIST Digit Recognition App")
st.write("Upload a handwritten digit image (0–9) and the CNN model will predict it.")

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_cnn_model()

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a digit image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"### ✅ Predicted Digit: **{predicted_digit}**")