import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Digit Recognition App", layout="centered")

st.title("🧠 MNIST Digit Recognition Web App")
st.write("Upload an image of a handwritten digit (0-9)")

MODEL_PATH = "model/mnist_cnn_model.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.write("Make sure your model is inside:")
    st.code("model/mnist_cnn_model.keras")
    st.stop()
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.success("✅ Model Loaded Successfully")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Resize to MNIST size
    image_resized = image.resize((28, 28))

    # Convert to numpy
    img_array = np.array(image_resized)

    # Invert if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display Results
    st.success(f"🎯 Predicted Digit: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.subheader("Prediction Probabilities")
    st.bar_chart(prediction[0])