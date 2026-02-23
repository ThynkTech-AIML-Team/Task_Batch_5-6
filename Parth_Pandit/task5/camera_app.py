import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Camera Digit Recognition")

@st.cache_resource
def load_cnn():
    return load_model("cnn_model.h5")

model = load_cnn()

st.title("Real-Time Digit Recognition (Camera)")
st.write("Show a handwritten digit (0–9) to the camera")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

        resized = cv2.resize(thresh, (28, 28))
        inverted = cv2.bitwise_not(resized)

        img = inverted / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        cv2.putText(
            frame,
            f"Prediction: {digit}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
else:
    st.write("Camera stopped")