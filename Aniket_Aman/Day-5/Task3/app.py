import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(page_title="Pro Digit AI", layout="centered")

# --- UI Header ---
st.title("🔢 Pro Digit Recognizer")
st.markdown("""
This model was trained with **Data Augmentation** and **Dropout** to be highly accurate.
Draw a digit below and see the prediction in real-time.
""")

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('mnist_model.keras')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model not found! Run the training script first.")
    st.stop()

# --- Canvas Logic ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Pad")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="pro_canvas",
    )

# --- Prediction Logic ---
if canvas_result.image_data is not None:
    # 1. Convert RGBA to Grayscale
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    
    # 2. Resize and Normalization
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_final = img_resized.reshape(1, 28, 28, 1) / 255.0

    with col2:
        st.subheader("AI Analysis")
        if st.button('Classify'):
            prediction = model.predict(img_final)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            st.write(f"### Predicted: **{digit}**")
            st.progress(float(confidence))
            st.write(f"Confidence: {confidence:.2%}")
            
            # Show the 28x28 version the model is actually seeing
            st.image(img_resized, caption="What the AI sees (28x28)", width=100)
            
            # Probability chart
            st.bar_chart(prediction[0])