import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("mnist_strong_cnn.h5")

st.title("🖊️ Digit Recognition App (MNIST)")

uploaded_file = st.file_uploader(
    "Upload a digit image (MNIST-like)", 
    type=["png", "jpg", "jpeg"]
)

def preprocess(img_bytes):
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    return img.reshape(1, 28, 28, 1), img

if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )

    X, display_img = preprocess(file_bytes)

    prediction = model.predict(X, verbose=0)
    digit_pred = np.argmax(prediction)

    st.image(display_img, caption="Processed Image (28×28)", width=150)
    st.success(f"Predicted Digit: {digit_pred}")
