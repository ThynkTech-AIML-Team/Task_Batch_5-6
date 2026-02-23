Image Processing and Deep Learning Projects

1. Image Classification Project

1.1 Objective

To build and compare image classification models using:

* Traditional Machine Learning (Logistic Regression)
* Deep Learning (Convolutional Neural Network)

Dataset used: MNIST Handwritten Digit Dataset.

 1.2 Dataset Description

* 70,000 grayscale images
* Image size: 28 × 28 pixels
* 10 classes (digits 0–9)
* 60,000 training images
* 10,000 testing images


 1.3 Preprocessing Steps

1. Normalization (pixel values scaled from 0–255 to 0–1)
2. Reshaping images for CNN input (28 × 28 × 1)
3. Flattening images for Logistic Regression (784 features)



1.4 Models Implemented

 1.4.1 Logistic Regression

* Flattened input
* Baseline traditional ML model

1.4.2 Convolutional Neural Network (CNN)

Architecture:

* Conv2D
* MaxPooling
* Flatten
* Dense (Hidden layer)
* Dense (Output layer with Softmax)

1.5 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Training vs Validation Accuracy Graph

1.6 Results

 Model                Accuracy 
 Logistic Regression  92–94%   
 CNN                  98–99%   


2. Image Processing Mini Tasks

2.1 Objective

To perform basic image processing operations using OpenCV.

2.2 Operations Performed

2.2.1 Edge Detection (Canny)

* Converts image to grayscale
* Detects object boundaries using gradient-based detection

2.2.2 Image Thresholding

* Converts grayscale image into binary image
* Separates foreground from background

2.2.3 Image Augmentation

* Horizontal Flip
* Rotation (45 degrees)
* Brightness Adjustment

2.3 Libraries Used

* OpenCV
* NumPy
* Matplotlib
* scikit-image (for sample image)

2.4 Outcome

Successfully demonstrated fundamental computer vision operations including feature extraction and image transformation.

3. Mini Deployment Project

3.1 Selected Option

Option A: Digit Recognition Web App using Streamlit

3.2 Objective

To deploy a trained MNIST CNN model as an interactive web application.

3.3 Application Features

1. User draws a digit (0–9) on a canvas.
2. Image is preprocessed automatically.
3. CNN model predicts the digit.
4. Predicted output is displayed instantly.

3.4 Technologies Used

* TensorFlow
* Streamlit
* streamlit-drawable-canvas
* NumPy

3.5 How to Run the Application

1. Install dependencies:

```
pip install tensorflow streamlit streamlit-drawable-canvas
```

2. Run the application:

```
streamlit run app.py
```

3. Open the local URL shown in the terminal.

---

3.6 Deployment Outcome

The model was successfully deployed as a web application capable of real-time digit prediction through user interaction.

