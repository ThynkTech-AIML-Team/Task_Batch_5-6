# MNIST Image Classification Project

## Objective
Build an image classification model using traditional ML and deep learning.

## Dataset
MNIST handwritten digits dataset (60,000 train, 10,000 test).

## Models Used
1. Logistic Regression
2. Convolutional Neural Network (CNN)

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~92% |
| CNN | ~98-99% |

## Evaluation
- Confusion Matrix
- Training vs Validation Accuracy Graph

## Image Processing Tasks
- Edge Detection (Canny)
- Thresholding
- Image Augmentation (Flip, Rotate, Brightness)

## Deployment
Streamlit Web App for Digit Recognition.

## How to Run
pip install -r requirements.txt
streamlit run app.py