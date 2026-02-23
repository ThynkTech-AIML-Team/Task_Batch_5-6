# CIFAR-10 Image Classifier Web Tool (Streamlit)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 1. Load Pre-trained CIFAR-10 Model
@st.cache_resource
def load_model():
    model_path = 'cnn_cifar10_model.h5'
    if not os.path.exists(model_path):
        st.warning('Trained model file cnn_cifar10_model.h5 not found. Please save your trained model in this directory.')
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Define Image Preprocessing Functions
def preprocess_image(image: Image.Image):
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32))
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    # If grayscale, convert to 3 channels
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    # If alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Build Image Upload Web Interface
st.title('CIFAR-10 Image Classifier')
st.write('Upload an image and the model will predict its class label (CIFAR-10 classes).')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

# 4. Predict Image Class Label
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image')
    img_array = preprocess_image(image)
    if model is not None:
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        st.write(f'**Predicted Class:** {pred_class}')
    else:
        st.warning('Model not loaded. Please ensure cnn_cifar10_model.h5 is present.')
