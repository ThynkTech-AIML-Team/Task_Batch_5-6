# Digit Recognition Web App & Image Processing Project

This project contains a complete end-to-end pipeline for image classification, specifically focusing on recognizing handwritten digits using the MNIST dataset, along with Jupyter notebooks for additional image processing tasks.

## Project Structure

The repository includes the following key components:

- **`app.py`**: A Streamlit web application that lets users draw a digit (0-9) on a canvas and get real-time predictions using the trained model.
- **`train_model.py`**: A Python script that downloads the MNIST dataset, builds a Convolutional Neural Network (CNN) using TensorFlow/Keras, and trains it to classify handwritten digits. The trained model is saved as `mnist_model.h5`.
- **`requirements.txt`**: A list of Python dependencies required to run the project.
- **`mnist_model.h5`**: The saved Keras model file generated after running `train_model.py`.
- **`1. Image Classification Project.ipynb`**: A Jupyter Notebook covering an image classification project workflow.
- **`2. Image Processing Mini Tasks.ipynb`**: A Jupyter Notebook detailing various mini-tasks related to image processing.

## Prerequisites

Ensure you have Python installed. The project recommends Python 3.12+ (tested with 3.13).

To install the required dependencies, run:

```bash
python -m pip install -r requirements.txt
```

## How to Run the Application

### 1. Train the Model
Before launching the web app, you must train the model and generate the `.h5` file.

```bash
python train_model.py
```

This will output `mnist_model.h5` in the root directory.

### 2. Start the Streamlit Web App
Launch the interactive web application to test the model:

```bash
python -m streamlit run app.py
```

The app will open in your default web browser (typically at `http://localhost:8502`). Draw a number on the black canvas, and click the **Predict** button to see the model's prediction and confidence score.

## Technologies Used
- [TensorFlow & Keras](https://www.tensorflow.org/) - For building and training the CNN model
- [Streamlit](https://streamlit.io/) - For the interactive web interface
- [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas) - For capturing user drawings
- [OpenCV](https://opencv.org/) - For image preprocessing (resizing, scaling, and conversion)

