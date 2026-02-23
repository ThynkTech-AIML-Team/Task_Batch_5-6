import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
import os

print(f"Using TensorFlow version: {tf.__version__}")
print("Downloading data and building model...")

# Load and prep data
(x_train, y_train), _ = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0

# Build model (Updated for modern Keras)
model = models.Sequential([
    Input(shape=(28, 28, 1)),  # <-- Explicit Input layer fixes the warning
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train for just 1 epoch to be fast
print("Training model... (This will take ~20 seconds)")
model.fit(x_train, y_train, epochs=1)

# Save the model using the modern .keras extension
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cnn_digit_model.keras")

model.save(MODEL_PATH)
print(f"✅ Model perfectly baked and saved at: {MODEL_PATH}")