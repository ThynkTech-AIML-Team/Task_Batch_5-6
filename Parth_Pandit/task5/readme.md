# Handwritten Digit Recognition using CNN (MNIST)

A deep learning project that recognizes handwritten digits using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The system supports **both mouse-drawn digit input and real-time camera-based digit recognition** using the same trained model.

---

## Features
- Digit recognition using a **mouse-drawn canvas**
- **Real-time webcam-based digit recognition**
- CNN trained on the MNIST dataset
- Fast and accurate predictions
- Single trained model used for both input methods

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Tkinter  

---

## Project Structure

├── model_training.ipynb # CNN training on MNIST dataset
├── app.py # Mouse-based digit drawing & prediction
├── camera_app.py # Webcam-based digit recognition
├── cnn_model.h5 # Trained CNN model
├── readme.md


---

## Dataset
- **MNIST Handwritten Digit Dataset**
- 60,000 training images
- 10,000 testing images
- 28×28 grayscale images of digits (0–9)

---

## Model Architecture
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected Dense layers
- Softmax output layer for multi-class classification

---

## Mouse-Based Digit Recognition
- User draws a digit using the mouse on a canvas
- Image is resized to **28×28**, normalized, and reshaped
- The trained CNN predicts the digit instantly

## Camera-Based Digit Recognition
- Uses webcam for real-time digit input
- Digit is extracted and preprocessed using OpenCV
- Preprocessed image is passed to the trained CNN for prediction

## Image Preprocessing Steps
- Grayscale conversion
- Noise removal and thresholding
- Image inversion (white digit on black background)
- Normalization (pixel values scaled to 0–1)
- Reshaping for CNN input

---

## How to Run the Project

1️.Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib

2.Train the Model (Optional)

Run the following notebook:

model_training.ipynb
3.Run Mouse-Based Digit Recognition
digit_draw_predict.ipynb
4.Run Camera-Based Digit Recognition
camera_digit_predict.ipynb