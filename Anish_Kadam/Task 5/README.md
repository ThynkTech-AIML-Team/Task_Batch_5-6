# Image Classification Project

## Objective
To build an image classification model using both traditional Machine Learning and Deep Learning techniques, and compare their performance.

---

## 📂 Dataset Used
Dataset: MNIST Digit Dataset  

- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 digit classes (0–9)
- 60,000 training images
- 10,000 testing images

---

## Project Workflow

### 1. Data Preprocessing
- Normalized pixel values (0–255 → 0–1)
- Reshaped images for CNN (28 × 28 × 1)
- Flattened images for Logistic Regression
- Split dataset into training and testing sets

---

### 2. Models Implemented

#### Logistic Regression (Traditional ML)
- Input: Flattened image vectors (784 features)
- Multi-class classification using softmax
- Baseline model for comparison

#### Convolutional Neural Network (CNN)
Architecture:
- Convolution Layer (ReLU)
- MaxPooling Layer
- Dropout
- Fully Connected Layer
- Softmax Output Layer

CNN automatically extracts spatial features and improves classification performance.

---

##  Evaluation Metrics

###  Accuracy
Measures percentage of correct predictions.

- Logistic Regression Accuracy:  0.9255 or ~92%
- CNN Accuracy: 0.9846000075340271 or ~98%

---

###  Confusion Matrix
Used to analyze:
- Correct classifications
- Misclassified digits
- Class-wise performance

CNN showed fewer misclassifications.

---

###  Training vs Validation Accuracy
- Plotted training and validation accuracy curves
- CNN converged smoothly
- Minimal overfitting observed

---

##  Model Comparison

| Model               | Accuracy | Performance |
|---------------------|----------|-------------|
| Logistic Regression | ~92%     | Good baseline |
| CNN                 | ~98%     | Excellent |

---

##  Conclusion
- CNN significantly outperformed Logistic Regression.
- Deep Learning models are better suited for image data.
- Traditional ML works but lacks spatial feature learning capability.


#  Image Processing Mini Tasks using OpenCV

##  Objective
To perform basic image processing operations using OpenCV and understand how image transformations work.

---

##  Tasks Performed

### 1. Edge Detection (Canny Edge Detection)
- Used OpenCV's Canny() function
- Converted image to grayscale
- Applied Gaussian Blur
- Detected edges using threshold values

Purpose:
Edge detection highlights object boundaries and important structural features in an image.

---

### 2. Image Thresholding
- Converted image to grayscale
- Applied thresholding using cv2.threshold()
- Generated binary image output

Purpose:
Thresholding simplifies the image by converting it into black and white based on pixel intensity.

---

### 3. Image Augmentation

#### a. Image Flipping
- Horizontal Flip

#### b. Image Rotation
- Rotated image 90 degrees using transformation matrix

#### c. Brightness Adjustment
- Increased brightness


---

##  Observations
- Edge detection clearly outlines shapes.
- Thresholding separates foreground and background.
- Augmentation techniques create variations of the same image.
- These techniques are useful in computer vision and deep learning pipelines.

---

##  Conclusion
This mini project demonstrates fundamental image processing techniques using OpenCV. These operations form the foundation of many computer vision and deep learning applications.

---

##  Applications
- Object Detection
- Medical Image Analysis
- Face Recognition
- Data Augmentation for Deep Learning