# 📌 Day-5: Computer Vision Project

## 👨‍💻 Internship Task – Task 5  
This project focuses on Image Classification, Image Processing using OpenCV, and Model Deployment using Streamlit.

---

# 🧠 1️⃣ Image Classification Project

## 🎯 Objective
Build an image classification model using:
- Traditional Machine Learning
- Neural Network (Deep Learning approach)

Dataset Used:
- MNIST Handwritten Digit Dataset

---

## 🔹 Steps Performed

### ✔ Data Preprocessing
- Normalization (pixel values scaled between 0–1)
- Train-test split
- Feature flattening for ML models

### ✔ Models Trained
1. Logistic Regression
2. Neural Network (MLPClassifier)

---

## 📊 Model Performance Comparison

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | ~92% |
| Neural Network (MLP) | ~97% |

### 📌 Observations:
- Neural Network outperformed Logistic Regression.
- Deep learning model captures complex patterns better.

---

## 📈 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Model Accuracy Comparison Graph

---

# 🖼 2️⃣ Image Processing Mini Tasks (OpenCV)

Performed the following operations:

### ✔ Edge Detection
- Canny Edge Detection

### ✔ Image Thresholding
- Binary Thresholding

### ✔ Image Augmentation
- Horizontal Flip
- Rotation (45 degrees)
- Brightness Adjustment


---

# 🌐 3️⃣ Mini Deployment Project

## 🧠 Digit Recognition Web App (Streamlit)

Built a web application where:
- User uploads handwritten digit image
- Image is resized to 28x28
- Model predicts digit
- Result displayed instantly

### ▶ Run App
```bash
cd app
streamlit run app.py