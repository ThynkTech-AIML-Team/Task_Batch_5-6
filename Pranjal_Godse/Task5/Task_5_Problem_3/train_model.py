import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def main():
    try:
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        print("Preprocessing data...")
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        X_train = X_train.reshape(-1,28,28,1)
        X_test = X_test.reshape(-1,28,28,1)

        print("Building model...")
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        print("Starting training (3 epochs)...")
        model.fit(X_train, y_train, epochs=3, verbose=2)

        out_path = "mnist_cnn_model.h5"
        print(f"Saving model to {out_path}...")
        model.save(out_path)

        print("Model saved. Directory listing:")
        for fname in sorted(os.listdir('.')):
            print(' -', fname)

        print("Model Saved Successfully!")

    except Exception as e:
        print("Error during training:", type(e).__name__, e)

if __name__ == '__main__':
    main()