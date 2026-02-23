# ============================================
# FINAL MNIST Digit Recognition App
# ============================================

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os


# ------------------------------
# Load model
# ------------------------------

if not os.path.exists("mnist_cnn_model.h5"):
    st.error("mnist_cnn_model.h5 not found in this folder.")
    st.stop()

model = load_model("mnist_cnn_model.h5")


# ------------------------------
# UI
# ------------------------------

st.title("MNIST Digit Recognition")
st.write("Upload a clear handwritten digit (black digit on white background)")


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)


# ------------------------------
# Preprocess Function (FIXED)
# ------------------------------

def preprocess(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find bounding box
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize digit to 20x20
    digit = cv2.resize(digit, (20, 20))

    # Add padding to make 28x28
    digit = np.pad(digit, ((4, 4), (4, 4)), "constant", constant_values=0)

    # Normalize
    digit = digit.astype("float32") / 255.0

    # Reshape
    digit = digit.reshape(1, 28, 28, 1)

    return digit


# ------------------------------
# Prediction
# ------------------------------

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", width=200)

    processed = preprocess(img)

    prediction = model.predict(processed)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(f"Predicted Digit: {digit}")

    st.info(f"Confidence: {confidence:.2f}")