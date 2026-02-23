IMAGE CLASSIFICATION PROJECT – MNIST DATASET

Objective:
The objective of this project is to build an image classification model using both traditional machine learning and deep learning techniques. The performance of the models is compared using evaluation metrics.

Dataset Used:
MNIST Digit Dataset

70,000 grayscale handwritten digit images

Image size: 28 x 28 pixels

10 classes (digits 0 to 9)

60,000 training images

10,000 testing images

Tasks Performed:

Image Preprocessing:

Converted images into numerical arrays

Resized images to 28 x 28 (if required)

Normalized pixel values from 0–255 to 0–1

Reshaped images to match model input

Flattened images for Logistic Regression

Train-Test Split:

Training data: 80%

Testing data: 20%
This ensures proper evaluation of the model.

Models Trained:

Logistic Regression:

Traditional machine learning model

Images flattened into 784 features (28 x 28)

Fast training

Used as baseline model

Accuracy: Around 92–94%

Limitation:
Does not capture spatial relationships between pixels.

Convolutional Neural Network (CNN):

Used Conv2D layers with ReLU activation

Used MaxPooling layers

Used Flatten layer

Used Dense layer

Used Softmax output layer

Advantage:
CNN captures spatial features and patterns in images.

Accuracy: Around 98–99%

Evaluation Metrics:

Accuracy:
Measures overall correctness of predictions.

Confusion Matrix:
Shows how many images were correctly and incorrectly classified for each digit.

Training vs Validation Accuracy Graph:
Used to check overfitting and underfitting.

Conclusion:
Logistic Regression provides good baseline performance.
CNN performs significantly better for image classification tasks.
Deep learning models are more suitable for image-based problems.