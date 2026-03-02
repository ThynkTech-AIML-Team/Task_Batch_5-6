# Image Processing Mini Tasks

This project demonstrates core image processing operations using **OpenCV** and **Python 3.11.1**.

## Tasks Performed
1.  **Edge Detection**: Implemented using the Canny algorithm.
2.  **Image Thresholding**: Compared Global, Otsu's, and Adaptive thresholding methods.
3.  **Image Augmentation**: Performed horizontal flip, 90-degree rotation, and brightness/darkness adjustments.

## Setup Instructions

1.  **Ensure Python 3.11.1 is installed.**
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    - Windows: `.\venv\Scripts\activate`
    - Linux/Mac: `source venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook image_processing_tasks.ipynb
    ```

## Technique Comparison Table

| Technique | Method | Accuracy/Quality | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Edge Detection** | Canny | High | Good noise reduction, accurate edge localization. | Requires manual tuning of two thresholds. |
| **Thresholding** | Global (Binary) | Low/Medium | Very fast and simple. | Not effective with non-uniform lighting. |
| **Thresholding** | Otsu's | Medium/High | Automatically calculates the best threshold value. | Still assumes bimodal distribution of intensity. |
| **Thresholding** | Adaptive | High | Handles different lighting conditions in different parts of the image. | More computationally expensive than global. |
| **Augmentation**| Flip/Rotate | Perfect | Essential for spatial invariance in models. | Increases training time if done on-the-fly. |
| **Augmentation**| Brightness | High | Simulates real-world lighting variations. | Can lose detail in extremely dark or bright areas. |

## Dependencies
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Jupyter Notebook
