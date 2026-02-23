import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

model = MobileNetV2(weights='imagenet')

st.title("🚀 Real-World Image Classifier")
st.write("Upload any image and get ImageNet predictions")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    image_resized = image.resize((224, 224))
    
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=5)[0]

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Top Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.write(f"{i+1}. {label} — {prob*100:.2f}%")
