import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from scipy import ndimage
import matplotlib.pyplot as plt


# Load Model
model = load_model("mnist_cnn_model.h5")

st.set_page_config(page_title="Digit Recognition", layout="centered")

st.title("🖊Digit Recognition")
st.write("Draw a digit clearly in the center")

# Session State
if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.current_index = 0


# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Preprocessing Function
def preprocess_image(img):

    # Convert RGBA → grayscale
    img = np.mean(img[:, :, :3], axis=2)

    # Smooth image (reduce noise)
    img = ndimage.gaussian_filter(img, sigma=1)

    # Threshold
    img = np.where(img > 50, 255, 0)

    # Find bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[y_min:y_max, x_min:x_max]

    # Make square
    height, width = img.shape
    size = max(height, width)
    square_img = np.zeros((size, size))
    square_img[
        (size - height)//2:(size - height)//2 + height,
        (size - width)//2:(size - width)//2 + width
    ] = img

    # Resize to 20x20 (MNIST style)
    img = Image.fromarray(square_img.astype('uint8')).resize((20, 20))
    img = np.array(img)

    # Add padding to make 28x28
    padded = np.zeros((28, 28))
    padded[4:24, 4:24] = img

    # Center using center of mass
    cy, cx = ndimage.center_of_mass(padded)
    shiftx = int(np.round(14 - cx))
    shifty = int(np.round(14 - cy))
    padded = ndimage.shift(padded, shift=[shifty, shiftx])

    # Invert colors
    padded = 255 - padded

    # Normalize
    padded = padded / 255.0

    # Reshape
    padded = padded.reshape(1, 28, 28, 1)

    return padded


# Predict Button
if st.button("Predict"):

    if canvas_result.image_data is not None:

        processed = preprocess_image(canvas_result.image_data)

        if processed is not None:

            prediction = model.predict(processed)[0]
            sorted_indices = np.argsort(prediction)[::-1]

            st.session_state.predictions = sorted_indices
            st.session_state.current_index = 0

        else:
            st.warning("Draw a clear digit.")


# Show Prediction
if st.session_state.predictions is not None:

    current_digit = st.session_state.predictions[
        st.session_state.current_index
    ]

    st.success(f"Predicted Digit: {current_digit}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Correct ✅"):
            st.success("Prediction Confirmed!")
            st.session_state.predictions = None

    with col2:
        if st.button("Wrong ❌"):
            if st.session_state.current_index < 9:
                st.session_state.current_index += 1
            else:
                st.warning("No more predictions available.")