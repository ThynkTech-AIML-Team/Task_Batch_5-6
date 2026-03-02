# Handwritten Digit Recognition Web App

Simple Streamlit app for predicting handwritten digits (0-9) using a trained MNIST CNN model.

## Project Files
- `app.py` - Streamlit web app (upload image and predict digit)
- `train_mnist_model.py` - Trains MNIST CNN and saves model
- `mnist_cnn.keras` - Trained model file used by app

## Requirements
Install dependencies:

```bash
pip install streamlit tensorflow opencv-python pillow numpy
```

## Run the App
From this folder:

```bash
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## If Model Is Missing
Train and create model file:

```bash
python train_mnist_model.py
```

This creates `mnist_cnn.keras` in the same directory.

## How to Use
1. Upload a digit image (`png`, `jpg`, `jpeg`, `bmp`).
2. App preprocesses image to `28x28` grayscale and normalizes to `0-1`.
3. App shows:
   - Predicted digit
   - Confidence percentage
   - Processed `28x28` image
   - Probability chart for digits `0-9`

## Notes
- Best results: white digit on dark/clean background.
- Restart Streamlit after replacing model file:

```bash
# stop app
Ctrl + C
# run again
streamlit run app.py
```
