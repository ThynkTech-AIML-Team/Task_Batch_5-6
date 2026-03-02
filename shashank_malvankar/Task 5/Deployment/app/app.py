import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/cnn_model.keras")

st.title("Image Classifier Web App")
st.write("Upload an image of a digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # Convert to numpy
    image = np.array(image)

    # Resize
    image = cv2.resize(image, (28, 28))

    # Threshold and invert
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Normalize
    image = image / 255.0

    # Reshape
    image = image.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.write("Prediction:", predicted_class)