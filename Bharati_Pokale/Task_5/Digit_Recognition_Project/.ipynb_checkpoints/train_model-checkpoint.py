{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "28336b8c-6bd9-4ec1-af6d-daeacce7ef4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m23s\u001b[0m 27ms/step - accuracy: 0.9431 - loss: 0.1941 - val_accuracy: 0.9759 - val_loss: 0.0855\n",
      "Epoch 2/5\n",
      "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 25ms/step - accuracy: 0.9821 - loss: 0.0585 - val_accuracy: 0.9858 - val_loss: 0.0472\n",
      "Epoch 3/5\n",
      "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 23ms/step - accuracy: 0.9880 - loss: 0.0393 - val_accuracy: 0.9863 - val_loss: 0.0464\n",
      "Epoch 4/5\n",
      "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 25ms/step - accuracy: 0.9904 - loss: 0.0306 - val_accuracy: 0.9885 - val_loss: 0.0391\n",
      "Epoch 5/5\n",
      "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 24ms/step - accuracy: 0.9923 - loss: 0.0239 - val_accuracy: 0.9877 - val_loss: 0.0386\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Saved Successfully!\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from tensorflow.keras.datasets import mnist\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# Load dataset\n",
    "(X_train, y_train), (X_test, y_test) = mnist.load_data()\n",
    "\n",
    "# Normalize\n",
    "X_train = X_train / 255.0\n",
    "X_test = X_test / 255.0\n",
    "\n",
    "# Reshape for CNN\n",
    "X_train = X_train.reshape(-1, 28, 28, 1)\n",
    "X_test = X_test.reshape(-1, 28, 28, 1)\n",
    "\n",
    "# One-hot encoding\n",
    "y_train = to_categorical(y_train)\n",
    "y_test = to_categorical(y_test)\n",
    "\n",
    "# Build CNN Model\n",
    "model = Sequential([\n",
    "    Input(shape=(28, 28, 1)),\n",
    "    Conv2D(32, (3,3), activation='relu'),\n",
    "    MaxPooling2D((2,2)),\n",
    "    Conv2D(64, (3,3), activation='relu'),\n",
    "    MaxPooling2D((2,2)),\n",
    "    Flatten(),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dense(10, activation='softmax')\n",
    "])\n",
    "\n",
    "# Compile model\n",
    "model.compile(optimizer='adam',\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "# Train model\n",
    "model.fit(X_train, y_train,\n",
    "          epochs=5,\n",
    "          validation_split=0.2,\n",
    "          batch_size=64)\n",
    "\n",
    "# Save model\n",
    "model.save(\"mnist_model.h5\")\n",
    "\n",
    "print(\"Model Saved Successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c35ab792-b4f1-455b-9f2c-374127b8c79f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
