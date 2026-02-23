import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
model = load_model("mnist_model.h5")

st.title("Digit Recognition Web App")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28,28))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1,28,28,1)

    prediction = model.predict(reshaped)
    predicted_digit = np.argmax(prediction)

    st.image(resized, caption="Processed Image", width=150)
    st.success(f"Predicted Digit: {predicted_digit}")