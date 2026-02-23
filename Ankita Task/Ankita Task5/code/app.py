import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("🧠 Simple Digit Recognition App")
st.write("Draw a digit (0-9):")

# -----------------------------
# Canvas for Drawing
# -----------------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Prediction Section
# -----------------------------
if canvas_result.image_data is not None:

    # Convert to numpy array
    img = canvas_result.image_data

    # Convert RGBA to grayscale
    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Resize to MNIST size
    img = cv2.resize(img, (28, 28))

    # Invert colors (MNIST format)
    img = 255 - img

    # Optional: Blur slightly to smooth strokes
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.subheader(f"Predicted Digit: {predicted_digit}")

    # Show processed image (debug view)
    st.image(img.reshape(28, 28), width=150, caption="Processed Image (28x28)")