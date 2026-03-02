# 🧠 Image Classification & Deployment Project

## 📌 Project Overview

This project demonstrates image classification using both traditional Machine Learning and Deep Learning techniques. It also includes image processing tasks using OpenCV and deployment of a trained model using Streamlit.

---

# 1️⃣ Image Classification Project

## 🎯 Objective

To build and compare image classification models using:

* Traditional Machine Learning (Logistic Regression)
* Deep Learning (Convolutional Neural Network - CNN)

The goal is to evaluate performance differences between classical ML and deep learning approaches on image data.

---

## 📂 Dataset Used

**MNIST Handwritten Digit Dataset**

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 (Grayscale)
* 10 classes (Digits 0–9)

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

* Resized images to 28 × 28 (if required)
* Normalized pixel values (0–255 → 0–1)
* Reshaped images for CNN input (28×28×1)
* Train-test split performed (already defined in MNIST)

---

## 🤖 Models Trained

### 1️⃣ Logistic Regression (Traditional ML)

* Images flattened into 784 features (28×28)
* Trained using scikit-learn
* Fast training but limited spatial understanding

### 2️⃣ Convolutional Neural Network (CNN)

* Conv2D layers for feature extraction
* MaxPooling for dimensionality reduction
* Fully connected dense layers for classification
* Achieves high accuracy due to spatial feature learning

---

## 📊 Model Performance Comparison

| Model               | Accuracy | Strengths                                | Limitations                     |
| ------------------- | -------- | ---------------------------------------- | ------------------------------- |
| Logistic Regression | ~92–94%  | Simple, fast                             | Cannot capture spatial patterns |
| CNN                 | ~98–99%  | High accuracy, spatial feature detection | Slightly higher training time   |

---

## 📈 Evaluation Metrics

### ✅ Accuracy

Used to measure overall prediction performance.

### ✅ Confusion Matrix

Shows class-wise prediction performance.

### ✅ Training vs Validation Accuracy Graph

Used to detect:

* Overfitting
* Underfitting
* Model convergence behavior

---

# 2️⃣ Image Processing Mini Tasks (OpenCV)

Implemented the following image processing techniques:

---

## 🔹 Edge Detection (Canny)

* Used `cv2.Canny()` to detect edges
* Helps in identifying object boundaries
* Commonly used in computer vision pipelines

---

## 🔹 Image Thresholding

* Applied binary thresholding
* Converts grayscale image to binary image
* Useful for segmentation tasks

---

## 🔹 Image Augmentation

Performed:

* Horizontal flip
* Rotation
* Brightness adjustment

Purpose:

* Increase dataset diversity
* Improve model generalization
* Reduce overfitting

---

# 3️⃣ Mini Deployment Project

## 🌐 Digit Recognition Web App (Streamlit)

A web application was built using Streamlit to:

* Draw handwritten digits
* Preprocess the image
* Predict digit using trained CNN model
* Display prediction result

---

## ⚙️ Deployment Steps

1. Train CNN model
2. Save model as `.h5`
3. Load model in `app.py`
4. Run using:

   ```
   streamlit run app.py
   ```

---

## 🖥️ Features of Web App

* Interactive drawing canvas
* Real-time prediction
* Image preprocessing (resize, normalize, invert)
* Processed image preview

---

# 📌 Key Learnings

* CNN significantly outperforms traditional ML in image tasks
* Proper preprocessing is critical for correct predictions
* Data augmentation improves robustness
* Model deployment bridges ML and real-world applications

---

# 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* OpenCV
* Streamlit
* NumPy
* Matplotlib

---

# 📊 Conclusion

This project demonstrates the transition from:

Traditional ML → Deep Learning → Deployment

CNN proved superior for image classification due to its ability to extract spatial features.

The deployed Streamlit app successfully performs real-time digit recognition using the trained model.

---

# 🚀 Future Improvements

* Use deeper CNN architecture
* Add probability visualization in web app
* Deploy on Streamlit Cloud
* Extend to CIFAR-10 dataset


