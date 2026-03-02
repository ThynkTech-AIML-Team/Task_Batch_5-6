import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

@st.cache_resource
def train_model():
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, verbose=0)
    return model

model = train_model()

st.title("🧠 MNIST Digit Recognition App")

st.write("Draw a digit (0–9) below:")

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


if canvas_result.image_data is not None:

    img = canvas_result.image_data

    # Convert to grayscale
    img = np.mean(img[:, :, :3], axis=2)

    # Invert (MNIST is white digit on black background)
    img = 255 - img

    # Resize to 28x28
    img = tf.image.resize(img[..., np.newaxis], (28, 28))
    img = img / 255.0
    img = img.numpy().reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    st.image(img.reshape(28,28), caption="Processed Image (28x28)")
    st.subheader(f"Predicted Digit: {predicted_digit}")