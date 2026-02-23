import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("🧠 MNIST Digit Recognition App")

# Load trained model
model = load_model("../notebooks/cnn_model.h5")

uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)

    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess
    image = cv2.resize(image,(28,28))
    image = image/255.0
    image = image.reshape(1,28,28,1)

    pred = np.argmax(model.predict(image),axis=1)[0]

    st.success(f"Predicted Digit: {pred}")