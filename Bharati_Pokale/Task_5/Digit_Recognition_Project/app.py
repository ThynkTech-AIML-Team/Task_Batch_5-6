import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title(" Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        
        # Convert image to array
        img = canvas_result.image_data
        
        # Convert RGBA to grayscale
        img = Image.fromarray((img).astype(np.uint8))
        img = img.convert("L")
        
        # Resize to 28x28
        img = img.resize((28,28))
        
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1,28,28,1)
        
        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        st.success(f"Predicted Digit: {predicted_class}")