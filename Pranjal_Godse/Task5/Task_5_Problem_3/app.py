import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🖼️ MNIST Image Classifier Web Tool")

uploaded_file = st.file_uploader("Upload a Digit Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    st.image(image, caption="Uploaded Image")
    st.success(f"Predicted Digit: {predicted_class}")