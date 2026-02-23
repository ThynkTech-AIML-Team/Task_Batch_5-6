1. Image Classification Project:
    -> This project focuses on building and comparing two image classification models:Logistic Regression (Traditional Machine Learning) 
    Convolutional Neural Network (Deep Learning)
    -> The goal is to evaluate performance differences between classical ML techniques and deep learning approaches for image classification

    -> Dataset used:
        MNIST handwritten digit dataset
        total images: 70,000
        classes: 10(0-9)

    -> Model comparison:
     logistic regression: Acc. - 92%
     CNN: Acc.- 99%
     
    -> evaluation matrix:
     Accuracy, confusion matrix, Training vs Validation Accuracy Graph 

2. Image Processing:
    operations performed:
    -> Load Image
    -> Resize Image
    -> Edge Detection
    -> thresholding

3. Model deployment using streamlit:
   -> This task deploys the trained deep learning model from Task 1 using Streamlit, allowing users to upload images and get predictions through a web interface.
   -> The model is trained on the MNIST dataset and deployed using Streamlit, allowing users to upload handwritten digit images and receive real-time predictions.

   Dataset used:
   -> MNIST handwritten digits
   -> is available from:
      from tensorflow.keras.datasets import mnist
 

    
