import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognition App")

@st.cache_resource
def load_cnn():
    return load_model("cnn_model.h5")

model = load_cnn()

st.title("Digit Recognition Web App")
st.write("Draw a digit (0–9) in the box below")

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

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0]  # grayscale
        resized = cv2.resize(img, (28, 28))
        inverted = cv2.bitwise_not(resized)
        normalized = inverted / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)

        prediction = model.predict(input_img)
        digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first")