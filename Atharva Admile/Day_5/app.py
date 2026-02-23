import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(
    page_title="Digit Recognition Web App",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Digit Recognition using MNIST")
st.markdown("Draw a digit (0-9) on the canvas below and click **Predict**.")

@st.cache_resource
def load_model():
    model_path = 'mnist_model.h5'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

if model is None:
    st.warning("Model not found. Please run `python train_model.py` first to generate `mnist_model.h5`.")
else:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="#000000",  # Fixed fill color with some opacity
        stroke_width=20,     # Stroke width for drawing clearly
        stroke_color="#FFFFFF", # White stroke (MNIST style - white digit on black background)
        background_color="#000000", # Black background
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # The canvas returns RGBA image
            # 1. Convert to Grayscale
            img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            
            # 2. Resize to 28x28 (MNIST input size)
            img_resized = cv2.resize(img, (28, 28))
            
            # 3. Normalize
            img_normalized = img_resized / 255.0
            
            # 4. Reshape to match model input shape (1, 28, 28, 1)
            img_reshaped = img_normalized.reshape(1, 28, 28, 1)
            
            # 5. Predict
            predictions = model.predict(img_reshaped)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Display results
            st.success(f"### Predicted Digit: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
            
            # Show the processed image for debugging
            st.markdown("Processed Input (28x28):")
            st.image(img_resized, width=100)
        else:
            st.error("Please draw a digit first.")
