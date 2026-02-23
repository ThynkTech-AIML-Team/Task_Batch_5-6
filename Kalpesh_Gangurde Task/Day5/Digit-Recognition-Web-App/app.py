import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_EXTENSIONS = ("*.keras", "*.h5")


def find_model_file(base_dir: Path) -> Path:
    model_files = []
    for pattern in MODEL_EXTENSIONS:
        model_files.extend(base_dir.glob(pattern))
    if not model_files:
        raise FileNotFoundError("No model file found. Add a .keras or .h5 model in this directory.")
    return sorted(model_files)[0]


@st.cache_resource
def load_mnist_model(model_path: Path):
    return load_model(model_path)


def preprocess_image_for_mnist(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    gray = image.convert("L")
    arr = np.array(gray)

    # Invert when background is light so drawn digit becomes bright foreground.
    if arr.mean() > 127:
        arr = 255 - arr

    resized = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    model_input = normalized.reshape(1, 28, 28, 1)
    return normalized, model_input


def predict_digit(model, model_input: np.ndarray) -> tuple[int, float, np.ndarray]:
    probs = model.predict(model_input, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100.0
    return digit, confidence, probs


def render_probability_chart(probabilities: np.ndarray) -> None:
    chart_data = {str(i): float(probabilities[i]) for i in range(10)}
    st.bar_chart(chart_data)


def main() -> None:
    st.set_page_config(page_title="Digit Recognition", page_icon=":1234:", layout="centered")
    st.title("Handwritten Digit Recognition")

    base_dir = Path(__file__).resolve().parent

    try:
        model_path = find_model_file(base_dir)
        model = load_mnist_model(model_path)
        st.caption(f"Loaded model: {model_path.name}")
    except Exception as exc:
        st.error(f"Model loading error: {exc}")
        st.stop()

    input_image = None

    uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        try:
            bytes_data = uploaded.read()
            input_image = Image.open(io.BytesIO(bytes_data))
        except Exception as exc:
            st.error(f"Image read error: {exc}")

    if input_image is None:
        st.info("Upload a digit image to get prediction.")
        st.stop()

    processed_28x28, model_input = preprocess_image_for_mnist(input_image)
    digit, confidence, probabilities = predict_digit(model, model_input)

    st.subheader("Prediction")
    st.success(f"Predicted digit: {digit}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Processed 28x28 Image")
    st.image(processed_28x28, clamp=True, channels="GRAY", width=220)

    st.subheader("Digit Probabilities (0-9)")
    render_probability_chart(probabilities)


if __name__ == "__main__":
    main()
