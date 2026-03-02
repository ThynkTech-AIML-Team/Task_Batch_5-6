MINI DEPLOYMENT PROJECT
Digit Recognition Web App

Objective:
To deploy a trained MNIST digit classification model as a web application using Streamlit, allowing users to draw handwritten digits and get real-time predictions.

Project Description:
In this project, a Convolutional Neural Network (CNN) model trained on the MNIST dataset is deployed using Streamlit. The web application provides an interactive interface where users can draw digits (0–9) on a canvas, and the trained model predicts the digit instantly.

Dataset Used:
MNIST Digit Dataset

70,000 grayscale handwritten digit images

Image size: 28 x 28 pixels

10 output classes (digits 0 to 9)

Model Used:
Convolutional Neural Network (CNN)

Architecture:

Conv2D layer with ReLU activation

MaxPooling layer

Conv2D layer with ReLU activation

MaxPooling layer

Flatten layer

Dense layer

Output layer with Softmax activation

Model Accuracy:
Approximately 98–99% on test data.

Application Workflow:

User draws a digit on the Streamlit canvas.

The drawn image is captured.

Image is converted to grayscale.

Image is resized to 28 x 28 pixels.

Pixel values are normalized (scaled between 0 and 1).

Image is reshaped to match the CNN input format.

The trained model predicts the digit.

The predicted result is displayed on the screen.

Technologies Used:

Python

Streamlit

TensorFlow / Keras

NumPy

OpenCV

streamlit-drawable-canvas

Features:

Interactive drawing interface

Real-time digit prediction

Fast and accurate model

Simple and user-friendly design