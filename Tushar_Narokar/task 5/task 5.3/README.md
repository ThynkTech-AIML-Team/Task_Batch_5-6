# Real-time Camera Prediction Project (High Accuracy)

This project implements a real-time object classification system using a webcam and deep learning models. It uses Python 3.11.1 and the **InceptionV3** architecture for high-accuracy real-time inference.

## Features
- Real-time video stream capture from webcam.
- Processing each frame through a pre-trained InceptionV3 model.
- Top-3 predictions displayed on the screen with confidence levels.
- Optimized for accuracy while maintaining usable frame rates on CPUs.

## Setup Instructions

1. **Create Virtual Environment:**
   ```bash
   py -3.11 -m venv venv
   ```

2. **Activate Virtual Environment:**
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python real_time_prediction.py
   ```

## Model Comparison

The following table compares different deep learning models for image classification:

| Model | Size (MB) | Parameters (M) | Top-1 Accuracy | Top-5 Accuracy | CPU Inference Speed (Approx) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | 14 | 3.5 | 71.3% | 90.1% | Fast |
| **ResNet50** | 98 | 25.6 | 74.9% | 92.1% | Moderate |
| **InceptionV3** | 92 | 23.9 | 77.9% | 93.7% | **Moderate (Selected for Accuracy)** |
| **VGG16** | 528 | 138.4 | 71.3% | 90.1% | Slow |
| **EfficientNetB0** | 29 | 5.3 | 77.1% | 93.3% | Fast |

### Chosen Model: InceptionV3
We upgraded the model from MobileNetV2 to **InceptionV3** to meet the user's request for higher accuracy. InceptionV3 offers a significant improvement in classification reliability (77.9% Top-1 accuracy) while still being efficient enough for real-time webcam inference on most modern processors.

## How it Works
1. **Webcam Capture:** OpenCV `VideoCapture` is used to access the camera hardware.
2. **Preprocessing:** Frames are resized to 299x299 pixels and normalized using `preprocess_input`.
3. **Inference:** The processed image is fed into the InceptionV3 model.
4. **Visualization:** The top 3 predicted labels and their probabilities are drawn on the original frame.

## Controls
- Press **'q'** to exit the application.
