# Image Classification Project

## Objective
The objective of this project is to build an image classification model using both traditional Machine Learning and Deep Learning techniques. The models are trained on the MNIST dataset to classify handwritten digits from 0 to 9.

---

## Dataset Used
MNIST Handwritten Digit Dataset

- Total images: 70,000
- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Classes: 10 (digits 0–9)

---

## Preprocessing Steps

- Loaded MNIST dataset using TensorFlow/Keras
- Normalized pixel values (0–255 → 0–1)
- Reshaped images for CNN input
- Flattened images for Logistic Regression

---

## Models Implemented

### 1. Logistic Regression
Traditional machine learning model trained on flattened image pixels.

Accuracy achieved: ~92%

---

### 2. Convolutional Neural Network (CNN)

Architecture:

- Conv2D Layer (32 filters)
- MaxPooling Layer
- Conv2D Layer (64 filters)
- MaxPooling Layer
- Flatten Layer
- Dense Layer (128 neurons)
- Output Layer (10 neurons, Softmax)

Accuracy achieved: ~98–99%

---

## Model Comparison

| Model | Accuracy |
|------|----------|
| Logistic Regression | ~92% |
| CNN | ~98–99% |

CNN performs significantly better due to automatic feature extraction.

---

## Evaluation Metrics Used

- Accuracy Score
- Confusion Matrix
- Training vs Validation Accuracy Graph

---

## Results

CNN achieved the highest accuracy and best performance.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---